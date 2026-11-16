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
    conventions of `PtyLab.Operators.Operators`.

    Examples:
        ```python
        import torch
        from PtyLab.Engines.GradientEngine.propagation import KernelPropagator

        propagate = KernelPropagator(reconstruction, "Fresnel", device="cpu")
        N = reconstruction.Np
        exit_wave = torch.ones(N, N, dtype=torch.complex64)
        propagate(exit_wave)  # detector field, same shape, centered order
        ```
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
        # Fraunhofer, Fresnel and scaledASP are built on the centered fft2c, so
        # their shifts can move to smaller tensors (see `unshifted`); ASP has none.
        self.centered = self.propagator != "asp"
        self.propagationKernels = self.unshiftedKernels = ()
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
        # The same kernels in unshifted FFT order, for `unshifted`. ASP's
        # transfer function is already in that order.
        self.unshiftedKernels = self.propagationKernels
        if self.centered:
            self.unshiftedKernels = tuple(
                torch.fft.ifftshift(kernel, dim=(-2, -1))
                for kernel in self.propagationKernels
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

    def unshifted(self, exit_wave):
        r"""
        Propagate an exit wave given in unshifted FFT order, without the centering shifts.

        With $S$ = `fftshift` and $S^{-1}$ = `ifftshift` over the last two
        axes, this computes $K$ in

        $$
        \mathcal{D}\lbrace\psi\rbrace = S\,K\lbrace S^{-1}\psi\rbrace,
        $$

        where $\mathcal{D}$ is `__call__`. With $\mathcal{F}$ the unshifted
        unitary FFT and $\tilde k = S^{-1}k$ a kernel in unshifted order,

        - Fraunhofer: $K\lbrace x\rbrace = \mathcal{F}\lbrace x\rbrace$,
        - Fresnel: $K\lbrace x\rbrace = \mathcal{F}\lbrace x\tilde Q\rbrace$,
        - ASP: $K\lbrace x\rbrace = \mathcal{F}^{-1}\lbrace \mathcal{F}\lbrace x\rbrace H\rbrace$,
          a circular convolution, which commutes with the shifts,
        - scaledASP:
          $K\lbrace x\rbrace = \mathcal{F}^{-1}\lbrace \mathcal{F}\lbrace x\tilde Q_s\rbrace \tilde H_s\rbrace$;
          the inner shifts of $\mathcal{F}_c^{-1}\mathcal{F}_c$ cancel.

        A shift only reorders pixels, so these identities are exact for even
        and odd grids. `PtychographyModel.forward` uses them to shift the
        object patches, the probe and the mode-summed intensity instead of
        every mode's exit wave and detector field.

        Args:
            exit_wave (torch.Tensor):
                Complex exit wave $S^{-1}\psi$ with spatial axes last.

        Returns:
            torch.Tensor:
                Detector field $S^{-1}\mathcal{D}\lbrace\psi\rbrace$ with the shape
                of `exit_wave`.
        """
        kernels = [kernel.to(dtype=exit_wave.dtype) for kernel in self.unshiftedKernels]
        if self.propagator == "fraunhofer":
            return torch.fft.fft2(exit_wave, norm="ortho")
        if self.propagator == "fresnel":
            return torch.fft.fft2(exit_wave * kernels[0], norm="ortho")
        if self.propagator == "asp":
            spectrum = torch.fft.fft2(exit_wave, norm="ortho")
            return torch.fft.ifft2(spectrum * kernels[0], norm="ortho")
        if self.propagator == "scaledasp":
            source_phase, transfer = kernels
            spectrum = torch.fft.fft2(exit_wave * source_phase, norm="ortho")
            return torch.fft.ifft2(spectrum * transfer, norm="ortho")
        raise NotImplementedError(f"Unsupported propagator: {self.propagator}")
