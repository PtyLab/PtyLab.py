"""Propagation of a Torch field, independent of optimization and monitoring."""

import numpy as np
import torch

from PtyLab.Operators._propagation_kernels import __make_quad_phase as make_quad_phase
from PtyLab.Operators.Operators import aspw, scaledASP


class KernelPropagator:
    """Callable with fixed geometry; preserves all leading field dimensions.

    Construct once per reconstruction, then call with a complex Torch exit wave.
    Kernels follow the existing PtyLab propagation conventions.
    """

    @staticmethod
    def fft2c(field):
        """Apply a centered, unitary 2D FFT over the last two dimensions.

        F_c(u) = fftshift(fft2(ifftshift(u))) / sqrt(H * W).
        H and W are the spatial sizes; leading batch/mode axes are preserved.
        """
        field = torch.fft.ifftshift(field, dim=(-2, -1))
        field = torch.fft.fft2(field, norm="ortho")
        return torch.fft.fftshift(field, dim=(-2, -1))

    @staticmethod
    def ifft2c(field):
        """Invert fft2c over the last two dimensions, preserving leading axes.

        F_c^{-1}(U) = fftshift(ifft2(ifftshift(U))) * sqrt(H * W),
        with H and W the spatial sizes and ifft2 in its default normalization.
        """
        field = torch.fft.ifftshift(field, dim=(-2, -1))
        field = torch.fft.ifft2(field, norm="ortho")
        return torch.fft.fftshift(field, dim=(-2, -1))

    def __init__(self, reconstruction, propagator, device="cpu"):
        """Build fixed geometry kernels once per run using PtyLab's conventions.

        No optimizable fields enter NumPy here. Subclasses learning geometry should
        instead build their kernels in Torch inside __call__(), on each forward
        pass, so those kernels remain part of the current autograd graph.
        """
        r = reconstruction
        self.propagator = propagator.lower()
        self.propagationKernels = ()
        if self.propagator == "fraunhofer":
            return
        if self.propagator in ("fresnel", "scaledasp") and r.zo == 0:
            raise ValueError(
                "Fresnel and scaledASP require nonzero propagation distance."
            )
        if self.propagator == "asp" and not np.isclose(r.dxp, r.dxd, rtol=1e-6, atol=0):
            raise ValueError(
                "ASP requires reconstruction.dxp == reconstruction.dxd; use scaledASP for different pixel spacings."
            )

        dummy = np.zeros((r.Np, r.Np), dtype=np.complex64)
        if self.propagator == "fresnel":
            kernels = (make_quad_phase(r.zo, r.wavelength, r.Np, r.dxp, False),)
        elif self.propagator == "asp":
            _, transfer = aspw(dummy, r.zo, r.wavelength, r.Lp)
            # Match propagate_ASP's unshifted FFT implementation, including odd grids.
            kernels = (np.fft.ifftshift(transfer.astype(np.complex64)),)
        elif self.propagator == "scaledasp":
            _, source_phase, transfer = scaledASP(
                dummy, r.zo, r.wavelength, r.dxo, r.dxd
            )
            kernels = (source_phase.astype(np.complex64), transfer.astype(np.complex64))
        else:
            raise NotImplementedError(
                "Provide a custom propagation callable for this propagator."
            )
        self.propagationKernels = tuple(
            torch.tensor(kernel, device=device) for kernel in kernels
        )

    def __call__(self, exit_wave):
        """Return a complex detector field with the exit wave's shape.

        exit_wave has spatial axes last and resides on the kernel device.
        Fixed kernels are cast to its dtype; gradients flow through the field.
        F_c below denotes fft2c and F denotes an unshifted unitary FFT.
        """
        kernels = [
            kernel.to(dtype=exit_wave.dtype) for kernel in self.propagationKernels
        ]
        if self.propagator == "fraunhofer":
            # u = F_c(psi), with psi the exit wave.
            return self.fft2c(exit_wave)
        if self.propagator == "fresnel":
            # u = F_c(psi * Q), using the prepared quadratic phase Q.
            return self.fft2c(exit_wave * kernels[0])
        if self.propagator == "asp":
            # u = F^{-1}(F(psi) * H), with H the angular-spectrum transfer.
            spectrum = torch.fft.fft2(exit_wave, norm="ortho")
            return torch.fft.ifft2(spectrum * kernels[0], norm="ortho")
        if self.propagator == "scaledasp":
            # u = F_c^{-1}(F_c(psi * Q_source) * H_scaled).
            source_phase, transfer = kernels
            return self.ifft2c(self.fft2c(exit_wave * source_phase) * transfer)
        raise NotImplementedError(f"Unsupported propagator: {self.propagator}")
