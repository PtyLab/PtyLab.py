"""
Generate synthetic CPM simulation data and save as simu.hdf5.

This module contains the simulation logic from example_scripts/simulateData.py
as a callable function with no side effects (no plotting, no os.chdir).
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import convolve2d

from PtyLab.Operators.Operators import aspw
from PtyLab.utils.scanGrids import GenerateNonUniformFermat
from PtyLab.utils.utils import cart2pol, circ, fft2c, gaussian2D


def generate_simu_hdf5(output_path: Path) -> None:
    """
    Generate a synthetic CPM ptychography dataset and save it as an HDF5 file.

    The simulation produces a focused Gaussian probe and a complex spiral object,
    computes diffraction patterns using Fraunhofer propagation, adds Poisson noise,
    and writes the result to *output_path*.

    Parameters
    ----------
    output_path : Path
        Destination file path (e.g. ``example_data/simu.hdf5``).
        The parent directory must already exist.
    """
    output_path = Path(output_path)

    # Physical properties
    wavelength = 632.8e-9
    zo = 5e-2
    binningFactor = 1

    # Detector coordinates
    Nd = 2**7
    dxd = 2**11 / Nd * 4.5e-6
    Ld = Nd * dxd

    # Probe coordinates
    dxp = wavelength * zo / Ld
    Np = Nd
    Lp = dxp * Np
    xp = np.arange(-Np // 2, Np // 2) * dxp
    Xp, Yp = np.meshgrid(xp, xp)

    # Object coordinates
    No = 2**10 + 2**9
    dxo = dxp
    Lo = dxo * No
    xo = np.arange(-No // 2, No // 2) * dxo
    Xo, Yo = np.meshgrid(xo, xo)

    # Generate illumination: focused beam via pinhole + lens
    f = 8e-3
    pinhole = circ(Xp, Yp, Lp / 2)
    pinhole = convolve2d(pinhole, gaussian2D(5, 1).astype(np.float32), mode="same")

    probe = aspw(pinhole, 2 * f, wavelength, Lp)[0]

    aperture = circ(Xp, Yp, 3 * Lp / 4)
    aperture = convolve2d(aperture, gaussian2D(5, 3).astype(np.float32), mode="same")
    probe = (
        probe
        * np.exp(-1.0j * 2 * np.pi / wavelength * (Xp**2 + Yp**2) / (2 * f))
        * aperture
    )
    probe = aspw(probe, 2 * f, wavelength, Lp)[0]

    # Generate object: complex spiral pattern
    d = 1e-3
    b = 33
    theta, rho = cart2pol(Xo, Yo)
    t = (1 + np.sign(np.sin(b * theta + 2 * np.pi * (rho / d) ** 2))) / 2
    phaseFun = np.exp(1.0j * (theta + 2 * np.pi * (rho / d) ** 2))
    t = t * circ(Xo, Yo, Lo) * (1 - circ(Xo, Yo, 200 * dxo)) * phaseFun + circ(
        Xo, Yo, 130 * dxo
    )
    obj = convolve2d(t, gaussian2D(5, 3), mode="same")
    object_ = obj * phaseFun

    # Generate scan positions (non-uniform Fermat spiral)
    numPoints = 100
    radius = 150
    p = 1
    R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)

    encoder = np.vstack((R * dxo, C * dxo)).T
    positions = np.round(encoder / dxo)
    offset = np.array([50, 20])
    positions = (positions + No // 2 - Np // 2 + offset).astype(int)
    numFrames = len(R)

    # Estimate beam size for entrancePupilDiameter
    beamSize = (
        np.sqrt(np.sum((Xp**2 + Yp**2) * np.abs(probe) ** 2) / np.sum(abs(probe) ** 2))
        * 2.355
    )

    # Compute ptychogram
    ptychogram = np.zeros((numFrames, Nd, Nd))
    for loop in np.arange(numFrames):
        row, col = positions[loop]
        sy = slice(row, row + Np)
        sx = slice(col, col + Np)
        objectPatch = object_[..., sy, sx].copy()
        esw = objectPatch * probe
        ESW = fft2c(esw)
        ptychogram[loop] = abs(ESW) ** 2

    # Simulate Poisson noise
    bitDepth = 14
    maxNumCountsPerDiff = 2**bitDepth
    ptychogram = ptychogram / np.max(ptychogram) * maxNumCountsPerDiff
    noise = np.random.poisson(ptychogram)
    ptychogram += noise
    ptychogram[ptychogram < 0] = 0

    # Save to HDF5
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
        hf.create_dataset("encoder", data=encoder, dtype="f")
        hf.create_dataset("binningFactor", data=binningFactor, dtype="i")
        hf.create_dataset("dxd", data=(dxd,), dtype="f")
        hf.create_dataset("Nd", data=(Nd,), dtype="i")
        hf.create_dataset("No", data=(No,), dtype="i")
        hf.create_dataset("zo", data=(zo,), dtype="f")
        hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
        hf.create_dataset("entrancePupilDiameter", data=(beamSize,), dtype="f")
        hf.create_dataset("orientation", data=0)
