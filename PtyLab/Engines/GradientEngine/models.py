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
    """

    def __init__(self, object_field, probe, positions):
        super().__init__()
        self.object = torch.nn.Parameter(
            torch.as_tensor(object_field, dtype=torch.complex64)
        )
        self.probe = probe
        self.register_buffer(
            "positions", torch.as_tensor(positions, dtype=torch.long), persistent=False
        )
        self.propagation = None

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

    def set_propagation(self, propagation):
        """Set the callable that maps exit waves to detector fields."""
        self.propagation = propagation

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

    def object_patches(self, indices):
        """Gather object patches with batch followed by the six physical axes."""
        positions = self.positions[indices]
        height, width = self.probe.field.shape[-2:]
        rows = (
            positions[:, 0, None, None]
            + torch.arange(height, device=self.object.device)[None, :, None]
        )
        cols = (
            positions[:, 1, None, None]
            + torch.arange(width, device=self.object.device)[None, None, :]
        )
        return self.object[..., rows, cols].movedim(-3, 0)

    def detector_fields(self, indices):
        """Return detector fields with all state axes retained."""
        if self.propagation is None:
            raise RuntimeError("Set model propagation before forwarding data.")
        patches = self.object_patches(indices)
        entrance_probe = self.probe(indices)
        return self.propagation(patches * entrance_probe)

    def forward(self, indices):
        """Return detector intensity with shape ``(batch, Np, Np)``."""
        fields = self.detector_fields(indices)
        return fields.abs().square().sum(dim=(1, 2, 3, 4))
