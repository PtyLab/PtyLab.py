"""
Torch components for the ptychographic forward model.
"""

import math

import torch


class SharedProbe(torch.nn.Module):
    r"""
    Entrance probe shared by every scan frame, stored scale-free.

    The physical probe is

    $$
    P = s\,F,
    $$

    where the field $F$ (`field`) is the optimized parameter and the scale $s$
    (`scale`) a fixed real buffer, which `GradientEngine` sets from the
    measured photons per frame (see `probe_scale`). $F$ is then of order one
    at every photon count, so one Adam learning rate fits all data: Adam steps
    are about `lr` in the parameter's own units.

    Args:
        field (array_like):
            Physical probe of shape `(nlambda, 1, npsm, 1, Np, Np)`.

        scale (float, optional):
            Probe scale $s$, rounded to a power of two. Defaults to 1.0.

    Notes:
        $s$ is kept a power of two, so converting between $F$ and the physical
        probe is exact in floating point.
    """

    def __init__(self, field, scale=1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(_power_of_two(scale)))
        self.field = torch.nn.Parameter(
            torch.as_tensor(field, dtype=torch.complex64) / self.scale
        )

    def physical_field(self):
        r"""
        Return the physical probe $P = sF$.

        Returns:
            torch.Tensor:
                Differentiable physical probe.
        """
        return self.scale * self.field

    def forward(self, indices):
        """
        Return one broadcast view of the six-axis physical probe per frame.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            torch.Tensor:
                Probe of shape `(batch, nlambda, 1, npsm, 1, Np, Np)`.
        """
        field = self.physical_field()
        return field.unsqueeze(0).expand(len(indices), *field.shape)

    def set_scale(self, scale):
        r"""
        Change the scale $s$ while keeping the physical probe unchanged.

        Args:
            scale (float):
                New scale, rounded to a power of two.
        """
        scale = _power_of_two(scale)
        with torch.no_grad():
            self.field.mul_(self.scale / scale)
            self.scale.fill_(scale)

    def reset_from_array(self, field):
        """
        Copy a physical probe into the existing parameter.

        Args:
            field (array_like):
                Physical probe with the shape of `field`.

        Raises:
            ValueError:
                If the shape differs from the parameter's.
        """
        value = torch.as_tensor(field, dtype=self.field.dtype, device=self.field.device)
        if value.shape != self.field.shape:
            raise ValueError(
                f"Probe shape changed from {tuple(self.field.shape)} to {tuple(value.shape)}."
            )
        with torch.no_grad():
            self.field.copy_(value / self.scale)


def probe_scale(photons_per_frame, frame_pixels):
    r"""
    Return the probe scale for a mean frame power, rounded to a power of two.

    $$
    s = \sqrt{\frac{N_{\mathrm{frame}}}{N_{\mathrm{pix}}}}
    $$

    is the amplitude of a probe spreading the frame's $N_{\mathrm{frame}}$
    photons evenly over the $N_{\mathrm{pix}}$ pixels of the probe window. A
    unitary propagator conserves power, so $\sum |P|^2$ is about the photons
    per frame for a transmission near one, and $|F|^2$ then averages about one
    over the window.

    Args:
        photons_per_frame (float):
            Mean measured photons per frame $N_{\mathrm{frame}}$.

        frame_pixels (int):
            Number of pixels $N_{\mathrm{pix}}$ in the probe window.

    Returns:
        float:
            Probe scale $s$, rounded to a power of two.

    Raises:
        ValueError:
            If the scale is not positive and finite.
    """
    return _power_of_two((float(photons_per_frame) / frame_pixels) ** 0.5)


def squared_modulus(field):
    r"""
    Return $|u|^2$ computed as $\mathrm{Re}(u)^2 + \mathrm{Im}(u)^2$.

    Args:
        field (torch.Tensor):
            Complex field $u$.

    Returns:
        torch.Tensor:
            Real squared modulus with the shape of `field`.

    Notes:
        `field.abs().square()` takes a square root only to square it again. On
        the 512/128 CPM fixture this form makes a full-scan CUDA iteration
        about 30% faster, with the same values and gradients up to rounding.
    """
    return torch.view_as_real(field).square().sum(dim=-1)


def _power_of_two(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Probe scale must be positive and finite.")
    return 2.0 ** round(math.log2(value))


class PtychographyModel(torch.nn.Module):
    r"""
    Single-slice CPM forward model with six-axis object and probe storage.

    For scan frame $j$ with patch origin $\mathbf{x}_j$, the predicted
    intensity is

    $$
    I_j = \sum_{\mathrm{modes}} \left|\mathcal{D}\left\lbrace O(\mathbf{r} + \mathbf{x}_j)\,P(\mathbf{r})\right\rbrace \right|^2 + b,
    $$

    where $\mathcal{D}$ is the propagation and $b$ the optional detector
    background.

    The object has shape `(nlambda, nosm, 1, nslice, No, No)` and the shared
    probe has shape `(nlambda, 1, npsm, 1, Np, Np)`. Stage 1 restricts all four
    physical counts to one while preserving these axes.

    `background` is an optional incoherent detector background added to every
    predicted frame, a scalar or one `(Np, Np)` image in the pixel order of
    `forward`'s output. It is an ordinary parameter, so it stays fixed unless
    selected for estimation.

    Args:
        object_field (array_like):
            Complex six-axis object.

        probe (SharedProbe):
            Probe module.

        positions (array_like):
            Integer patch origins of shape `(frames, 2)` in object pixels.
    """

    def __init__(self, object_field, probe, positions):
        super().__init__()
        self.object = torch.nn.Parameter(
            torch.as_tensor(object_field, dtype=torch.complex64)
        )
        self.probe = probe
        # Flat object offsets of one probe-sized patch, fixed by the two grids.
        height, width = probe.field.shape[-2:]
        offsets = (
            torch.arange(height)[:, None] * self.object.shape[-1]
            + torch.arange(width)[None, :]
        )
        self.register_buffer("patch_offsets", offsets, persistent=False)
        self.register_buffer("positions", None, persistent=False)
        self.register_buffer("patch_origins", None, persistent=False)
        self.register_parameter("background", None)
        self.set_positions(positions)
        self.propagation = None
        self.intensity_propagation = None

    @staticmethod
    def validate(reconstruction):
        """
        Validate the Stage 1 counts and the canonical six-axis array shapes.

        Args:
            reconstruction (Reconstruction):
                Reconstruction state to check.

        Raises:
            NotImplementedError:
                If there is more than one wavelength, object state, probe state
                or slice.

            ValueError:
                If the object or probe is not a six-axis array.
        """
        recon = reconstruction
        if any(
            getattr(recon, name) != 1 for name in ("nlambda", "nosm", "npsm", "nslice")
        ):
            raise NotImplementedError(
                "PtychographyModel currently requires one wavelength, object state, "
                "probe state, and slice."
            )
        object_shape = (1, 1, 1, 1, recon.No, recon.No)
        probe_shape = (1, 1, 1, 1, recon.Np, recon.Np)
        if recon.object.shape != object_shape or recon.probe.shape != probe_shape:
            raise ValueError(
                "Initialize six-axis object and probe arrays with initializeObjectProbe()."
            )

    def set_positions(self, positions):
        """
        Replace the integer patch origins without changing model parameters.

        Args:
            positions (array_like):
                Patch origins of shape `(frames, 2)` in object pixels.
        """
        self.positions = torch.as_tensor(
            positions, dtype=torch.long, device=self.object.device
        )
        # Flat index of each patch's top-left pixel, cached for every gather.
        self.patch_origins = (
            self.positions[:, 0] * self.object.shape[-1] + self.positions[:, 1]
        )

    def set_propagation(self, propagation, intensity_propagation=None):
        """
        Set the callables that map exit waves to detector fields.

        Args:
            propagation (callable):
                Maps exit waves to complex detector fields. `detector_fields`
                always uses it.

            intensity_propagation (callable, optional):
                Only has to preserve the squared modulus up to a pixel order,
                like `KernelPropagator.intensity_field`. `forward` uses it when
                given.
        """
        self.propagation = propagation
        self.intensity_propagation = intensity_propagation

    def set_background(self, background):
        """
        Replace the detector background, or remove it.

        Args:
            background (float or array_like or None):
                Real scalar or one-frame image in `forward`'s pixel order,
                copied to the object's device as float32. None removes it.
        """
        if background is None:
            self.background = None
            return
        self.background = torch.nn.Parameter(
            torch.as_tensor(background, dtype=torch.float32, device=self.object.device)
            .detach()
            .clone()
        )

    def reset_from_reconstruction(self, reconstruction):
        """
        Import object, probe and positions while preserving parameter identity.

        Args:
            reconstruction (Reconstruction):
                Source of the object, probe and positions.

        Raises:
            ValueError:
                If the object or probe shape changed.
        """
        obj = torch.as_tensor(
            reconstruction.object, dtype=self.object.dtype, device=self.object.device
        )
        if obj.shape != self.object.shape:
            raise ValueError(
                f"Object shape changed from {tuple(self.object.shape)} to {tuple(obj.shape)}."
            )
        with torch.no_grad():
            self.object.copy_(obj)
        self.probe.reset_from_array(reconstruction.probe)
        self.set_positions(reconstruction.positions)

    def patch_indices(self, indices):
        """
        Return the flat object indices of the patches of a batch.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            torch.Tensor:
                Flat indices of shape `(batch, Np, Np)`.
        """
        return self.patch_origins[indices, None, None] + self.patch_offsets

    def object_patches(self, indices):
        """
        Gather the object patches of a batch.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            torch.Tensor:
                Patches with the batch axis followed by the six physical axes.
        """
        flat = self.patch_indices(indices)
        patches = self.object.flatten(-2).index_select(-1, flat.reshape(-1))
        return patches.unflatten(-1, flat.shape).movedim(-3, 0)

    def gradient_preconditioners(self, indices):
        r"""
        Return diagonal curvature maps for PIE-style preconditioning.

        $$
        D_O(\mathbf{x}) = \sum_{j} |P(\mathbf{x} - \mathbf{x}_j)|^2,\qquad D_F(\mathbf{r}) = s^2 \sum_{j} |O(\mathbf{r} + \mathbf{x}_j)|^2,
        $$

        summed over the frames $j$ in `indices`. These are the Gauss-Newton
        diagonals for a loss with constant curvature in the detector field, as
        used by PIE-type updates. Both are for the stored parameters: the probe
        map carries the $s^2$ of a scale-free probe, so a preconditioned step
        moves the physical probe independently of $s$.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            dict:
                `"object"`: $D_O$ with the object's shape, and
                `"probe.field"`: $D_F$.
        """
        with torch.no_grad():
            probe_intensity = self.probe(indices).abs().square()
            probe_intensity = probe_intensity.sum(dim=(1, 2, 3, 4))
            illumination = torch.zeros(
                self.object.shape[-2:].numel(),
                dtype=probe_intensity.dtype,
                device=self.object.device,
            )
            illumination.index_add_(
                0, self.patch_indices(indices).reshape(-1), probe_intensity.reshape(-1)
            )
            illumination = illumination.view(self.object.shape[-2:])
            coverage = self.object_patches(indices).abs().square().sum(dim=0)
            coverage = coverage * getattr(self.probe, "scale", 1.0) ** 2
        return {
            "object": illumination.expand(self.object.shape),
            "probe.field": coverage,
        }

    def detector_fields(self, indices):
        """
        Return the complex detector fields of a batch.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            torch.Tensor:
                Detector fields with all state axes retained.

        Raises:
            RuntimeError:
                If no propagation is set.
        """
        if self.propagation is None:
            raise RuntimeError("Set model propagation before forwarding data.")
        patches = self.object_patches(indices)
        entrance_probe = self.probe(indices)
        return self.propagation(patches * entrance_probe)

    def forward(self, indices):
        """
        Return the predicted detector intensity of a batch, plus any background.

        Args:
            indices (torch.Tensor):
                Frame indices of the batch.

        Returns:
            torch.Tensor:
                Intensity of shape `(batch, Np, Np)`, in the pixel order of
                `intensity_propagation` when one is set.
        """
        if self.intensity_propagation is None:
            fields = self.detector_fields(indices)
        else:
            fields = self.intensity_propagation(
                self.object_patches(indices) * self.probe(indices)
            )
        intensity = squared_modulus(fields).sum(dim=(1, 2, 3, 4))
        if self.background is not None:
            intensity = intensity + self.background
        return intensity
