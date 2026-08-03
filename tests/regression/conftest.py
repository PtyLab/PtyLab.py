"""Fixtures for the numerical regression suite.

The datasets here are synthesized in-test with a fixed seed rather than loaded
from ``example_data/``. That is deliberate: locally ``example_data/simu.hdf5``
is a real 100x128x128 measurement, while CI synthesizes a 20x64x64 stand-in, so
a golden recorded against "simu.hdf5" would not be reproducible across the two.
Everything the goldens depend on is generated here.
"""

import h5py
import numpy as np
import pytest

# Small enough that a handful of iterations of every engine runs in ~seconds,
# large enough that the object is meaningfully bigger than the probe (so the
# patch-extraction and scatter-back indexing is actually exercised).
ND = 32
GRID = 4  # GRID x GRID raster scan -> 16 frames
SEED = 20240607


def _simulate_dataset(path, nd=ND, grid=GRID, seed=SEED):
    """Write a deterministic CPM dataset to ``path``."""
    rng = np.random.default_rng(seed)
    n_frames = grid * grid

    # A raster scan, in metres, with a slight jitter so positions are not
    # perfectly degenerate.
    step = 3e-6
    coords = (np.arange(grid) - (grid - 1) / 2) * step
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    encoder = np.stack([yy.ravel(), xx.ravel()], axis=1)
    encoder = encoder + rng.normal(0, step / 50, encoder.shape)

    # Smooth-ish speckle: random field low-pass filtered in Fourier space, so
    # the diffraction patterns look like data rather than white noise.
    field = rng.random((n_frames, nd, nd))
    spectrum = np.fft.fftshift(np.fft.fft2(field), axes=(-2, -1))
    ky = np.fft.fftshift(np.fft.fftfreq(nd))
    mask = (ky[:, None] ** 2 + ky[None, :] ** 2) < 0.25**2
    ptychogram = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum * mask, axes=(-2, -1)))) ** 2
    ptychogram = (ptychogram / ptychogram.max()).astype(np.float32)

    with h5py.File(path, "w") as hf:
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
        hf.create_dataset("encoder", data=encoder, dtype="f")
        hf.create_dataset("dxd", data=np.array(75e-6))
        hf.create_dataset("zo", data=np.array(0.05))
        hf.create_dataset("wavelength", data=np.array(632.8e-9))
        hf.create_dataset("entrancePupilDiameter", data=np.array(400e-6))
    return path


@pytest.fixture(scope="session")
def regression_dataset(tmp_path_factory):
    """Path to a deterministic synthetic CPM dataset."""
    path = tmp_path_factory.mktemp("regression_data") / "regression_cpm.hdf5"
    return _simulate_dataset(path)
