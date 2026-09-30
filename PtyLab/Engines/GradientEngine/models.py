"""Torch components for the ptychographic forward model."""

import torch


class SharedProbe(torch.nn.Module):
    """An entrance probe shared by every scan frame."""

    def __init__(self, field):
        super().__init__()
        self.field = torch.nn.Parameter(torch.as_tensor(field, dtype=torch.complex64))

    def forward(self, indices):
        """Return one broadcast view of the six-axis probe per frame index."""
        return self.field.unsqueeze(0).expand(len(indices), *self.field.shape)

    def reset_from_array(self, field):
        """Copy a reconstruction probe into the existing parameter."""
        value = torch.as_tensor(field, dtype=self.field.dtype, device=self.field.device)
        if value.shape != self.field.shape:
            raise ValueError(
                f"Probe shape changed from {tuple(self.field.shape)} to {tuple(value.shape)}."
            )
        with torch.no_grad():
            self.field.copy_(value)


class PtychographyModel(torch.nn.Module):
    """Single-slice CPM components with six-axis object/probe storage.

    ``object`` has shape ``(nlambda, nosm, 1, nslice, No, No)`` and the
    shared probe has shape ``(nlambda, 1, npsm, 1, Np, Np)``. Stage 1
    restricts all four physical counts to one while preserving these axes.

    ``background`` is an optional incoherent detector background added to every
    predicted frame, a scalar or one ``(Np, Np)`` image in the same pixel order
    as ``forward``'s output. It is an ordinary parameter, so it stays fixed
    unless selected for estimation.
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
        """Validate Stage 1 counts and the canonical six-axis array shapes."""
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
        """Replace integer patch origins without changing model parameters."""
        self.positions = torch.as_tensor(
            positions, dtype=torch.long, device=self.object.device
        )
        # Flat index of each patch's top-left pixel, cached for every gather.
        self.patch_origins = (
            self.positions[:, 0] * self.object.shape[-1] + self.positions[:, 1]
        )

    def set_propagation(self, propagation, intensity_propagation=None):
        """Set the callables that map exit waves to detector fields.

        ``intensity_propagation`` only has to preserve the squared modulus up to
        a pixel order, like ``KernelPropagator.intensity_field``; ``forward``
        uses it when given and ``detector_fields`` always uses ``propagation``.
        """
        self.propagation = propagation
        self.intensity_propagation = intensity_propagation

    def set_background(self, background):
        """Replace the detector background, or remove it with ``None``.

        ``background`` is a real scalar or one-frame tensor in ``forward``'s
        pixel order; it is copied to the object's device as float32.
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
        """Import object and probe values while preserving parameter identity."""
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
        """Return flat object indices with shape ``(batch, Np, Np)``."""
        return self.patch_origins[indices, None, None] + self.patch_offsets

    def object_patches(self, indices):
        """Gather object patches with batch followed by the six physical axes."""
        flat = self.patch_indices(indices)
        patches = self.object.flatten(-2).index_select(-1, flat.reshape(-1))
        return patches.unflatten(-1, flat.shape).movedim(-3, 0)

    def gradient_preconditioners(self, indices):
        """Return diagonal curvature maps for PIE-style preconditioning.

        ``object`` receives the probe intensity summed over the patches of
        ``indices``, and ``probe.field`` the summed object-patch intensity.
        These are the Gauss-Newton diagonals for a loss with constant curvature
        in the detector field, as used by PIE-type updates.
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
        return {
            "object": illumination.expand(self.object.shape),
            "probe.field": coverage,
        }

    def detector_fields(self, indices):
        """Return detector fields with all state axes retained."""
        if self.propagation is None:
            raise RuntimeError("Set model propagation before forwarding data.")
        patches = self.object_patches(indices)
        entrance_probe = self.probe(indices)
        return self.propagation(patches * entrance_probe)

    def forward(self, indices):
        """Return detector intensity, plus any background, as ``(batch, Np, Np)``.

        The pixel order is that of ``intensity_propagation`` when one is set.
        """
        if self.intensity_propagation is None:
            fields = self.detector_fields(indices)
        else:
            fields = self.intensity_propagation(
                self.object_patches(indices) * self.probe(indices)
            )
        intensity = fields.abs().square().sum(dim=(1, 2, 3, 4))
        if self.background is not None:
            intensity = intensity + self.background
        return intensity
