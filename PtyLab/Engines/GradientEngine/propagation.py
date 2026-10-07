"""
Propagation of a Torch field, independent of optimization and monitoring.
"""

import numpy as np
import torch

from PtyLab.Operators._propagation_kernels import __make_quad_phase as make_quad_phase
from PtyLab.Operators.Operators import aspw, scaledASP


class KernelPropagator:
    """
    Propagate exit waves to the detector with fixed geometry.

    Construct once per reconstruction, then call with a complex Torch exit wave.
    All leading field dimensions are preserved, and the kernels follow the
    existing PtyLab propagation conventions.
    """

    @staticmethod
    def fft2c(field):
        r"""
        Apply a centered, unitary 2D FFT over the last two dimensions.

        $$
        \mathcal{F}_c\lbrace u\rbrace = \frac{1}{\sqrt{HW}}\, \mathrm{fftshift}\left(\mathrm{fft2}\left(\mathrm{ifftshift}(u)\right)\right)
        $$

        where $H$ and $W$ are the spatial sizes. Leading batch and mode axes are
        preserved.

        Args:
            field (torch.Tensor):
                Complex field with spatial axes last.

        Returns:
            torch.Tensor:
                Centered spectrum with the shape of `field`.
        """
        field = torch.fft.ifftshift(field, dim=(-2, -1))
        field = torch.fft.fft2(field, norm="ortho")
        return torch.fft.fftshift(field, dim=(-2, -1))

    @staticmethod
    def ifft2c(field):
        r"""
        Invert `fft2c` over the last two dimensions.

        $$
        \mathcal{F}_c^{-1}\lbrace U\rbrace = \sqrt{HW}\, \mathrm{fftshift}\left(\mathrm{ifft2}\left(\mathrm{ifftshift}(U)\right)\right)
        $$

        where $H$ and $W$ are the spatial sizes and $\mathrm{ifft2}$ has its
        default normalization. Leading axes are preserved.

        Args:
            field (torch.Tensor):
                Complex centered spectrum with spatial axes last.

        Returns:
            torch.Tensor:
                Field with the shape of `field`.
        """
        field = torch.fft.ifftshift(field, dim=(-2, -1))
        field = torch.fft.ifft2(field, norm="ortho")
        return torch.fft.fftshift(field, dim=(-2, -1))

    def __init__(self, reconstruction, propagator, device="cpu"):
        """
        Build the fixed geometry kernels once per run.

        No optimizable fields enter NumPy here. Subclasses that learn the
        geometry should instead build their kernels in Torch inside
        `__call__()`, on each forward pass, so those kernels remain part of the
        current autograd graph.

        Args:
            reconstruction (Reconstruction):
                Reconstruction state providing the geometry (`Np`, `dxp`,
                `dxd`, `dxo`, `zo`, `wavelength`, `Lp`).

            propagator (str):
                `"Fraunhofer"`, `"Fresnel"`, `"ASP"` or `"scaledASP"`, in any
                case.

            device (str or torch.device, optional):
                Device of the kernels. Defaults to `"cpu"`.

        Raises:
            ValueError:
                If Fresnel or scaledASP has zero propagation distance, or ASP
                has different object and detector pixel sizes.

            NotImplementedError:
                For any other propagator.
        """
        r = reconstruction
        self.propagator = propagator.lower()
        self.propagationKernels = ()
        # intensity_field() skips both centering shifts, see its docstring.
        self.fftOrderIntensity = self.propagator == "fraunhofer"
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
        r"""
        Propagate an exit wave $\psi$ to the detector.

        With $\mathcal{F}_c$ the centered and $\mathcal{F}$ the unshifted
        unitary FFT, the detector field is

        - Fraunhofer: $u = \mathcal{F}_c\lbrace \psi\rbrace$,
        - Fresnel: $u = \mathcal{F}_c\lbrace \psi Q\rbrace$, with the quadratic phase $Q$,
        - ASP: $u = \mathcal{F}^{-1}\lbrace \mathcal{F}\lbrace \psi\rbrace H\rbrace$, with the
          angular-spectrum transfer function $H$,
        - scaledASP:
          $u = \mathcal{F}_c^{-1}\lbrace \mathcal{F}_c\lbrace \psi Q_s\rbrace H_s\rbrace$,
          with the source phase $Q_s$ and scaled transfer function $H_s$.

        Fixed kernels are cast to the field's dtype; gradients flow through
        the field.

        Args:
            exit_wave (torch.Tensor):
                Complex exit wave with spatial axes last, on the kernel device.

        Returns:
            torch.Tensor:
                Complex detector field with the shape of `exit_wave`.
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

    def intensity_field(self, exit_wave):
        r"""
        Return a field whose squared modulus is the detector intensity.

        With `fftOrderIntensity` (Fraunhofer) the intensity is in unshifted FFT
        order and compares with `ifftshift` of centered data. The plain
        $\mathcal{F}\lbrace \psi\rbrace$ is used: the input `ifftshift` of
        $\mathcal{F}_c$ is a linear far-field phase and the output `fftshift` a
        pixel permutation, so

        $$
        |\mathcal{F}_c\lbrace \psi\rbrace |^2 = \mathrm{fftshift}\left(|\mathcal{F}\lbrace \psi\rbrace |^2\right),
        $$

        also for odd grids. Other propagators return the centered field.

        Args:
            exit_wave (torch.Tensor):
                Complex exit wave with spatial axes last.

        Returns:
            torch.Tensor:
                Complex detector field.
        """
        if self.fftOrderIntensity:
            return torch.fft.fft2(exit_wave, norm="ortho")
        return self(exit_wave)
