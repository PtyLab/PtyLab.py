"""Measurement models: object/probe tensors and a scan position to a detector field.

Models contain the physics, not the optimizer or NumPy monitoring state. The
propagate callback accepts an object patch and probe and returns a Torch field.
"""


class SingleSliceModel:
    """Single-wavelength, single-mode, single slice CPM with fixed integer scan positions.

    For scan j, the detector field is u_j = D[O_j * P], where O_j is
    the object patch at that position, P is the probe, and D propagates
    their pointwise product to the detector.
    """

    def validate(self, reconstruction):
        """Require one wavelength, mode and slice, with six-dimensional arrays.

        Raise NotImplementedError for unsupported mode counts and ValueError
        for object/probe shapes inconsistent with the initialized grid sizes.
        """
        r = reconstruction
        if any(getattr(r, name) != 1 for name in ("nlambda", "nosm", "npsm", "nslice")):
            raise NotImplementedError(
                "ObjectProbeModel requires single-mode, single-slice data."
            )
        object_shape = (1, 1, 1, 1, r.No, r.No)
        probe_shape = (1, 1, 1, 1, r.Np, r.Np)
        if r.object.shape != object_shape or r.probe.shape != probe_shape:
            raise ValueError(
                "Initialize single-mode object and probe with initializeObjectProbe()."
            )

    def __call__(self, obj, probe, position, propagate):
        """Return a [height, width] complex detector field with gradients intact.

        position gives the integer (row, col) of the object's patch origin.
        propagate receives that patch and the probe as separate tensors.
        """
        row, col = position
        height, width = probe.shape[-2:]
        # O_j[y, x] = O[row_j + y, col_j + x].
        patch = obj[..., row : row + height, col : col + width]
        return propagate(patch, probe).reshape(height, width)
