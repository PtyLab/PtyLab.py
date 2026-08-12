import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction


@pytest.fixture
def fpm_reconstruction(tmp_path):
    """A minimal FPM dataset, sampled like the LED-array microscope examples."""
    rng = np.random.default_rng(42)
    Nd, N_frames = 64, 25

    # LEDs on a 5x5 grid, 60 mm below the sample
    led = np.linspace(-4e-3, 4e-3, 5)
    encoder = np.stack(np.meshgrid(led, led), axis=-1).reshape(-1, 2)

    hdf5_path = tmp_path / "fpm.hdf5"
    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset(
            "ptychogram", data=rng.random((N_frames, Nd, Nd)).astype(np.float32)
        )
        hf.create_dataset("encoder", data=encoder)
        hf.create_dataset("dxd", data=np.array(5.5e-6))
        hf.create_dataset("magnification", data=np.array(4.0))
        hf.create_dataset("wavelength", data=np.array(625e-9))
        hf.create_dataset("zled", data=np.array(60e-3))
        hf.create_dataset("NA", data=np.array(0.1))

    data = ExperimentalData(hdf5_path, operationMode="FPM")
    return Reconstruction(data, Params())


def test_fpm_object_sampling_preserves_field_of_view(fpm_reconstruction):
    """The enlarged FPM object adds bandwidth, not field of view."""
    reconstruction = fpm_reconstruction
    assert reconstruction.No > reconstruction.Np

    assert_allclose(
        reconstruction.dxo_fpm,
        reconstruction.dxp * reconstruction.Np / reconstruction.No,
    )
    assert reconstruction.dxo_fpm < reconstruction.dxp
    assert_allclose(reconstruction.Lo_fpm, reconstruction.Np * reconstruction.dxp)


def test_monitor_plots_fpm_object_with_fpm_pixel_size(fpm_reconstruction):
    monitor = Monitor()
    monitor.reconstruction = fpm_reconstruction

    assert_allclose(monitor.objectPixelSize, fpm_reconstruction.dxo_fpm)
    # the plotted extent is the field of view of the raw images, not No * dxo
    assert_allclose(
        fpm_reconstruction.No * monitor.objectPixelSize,
        fpm_reconstruction.Np * fpm_reconstruction.dxp,
    )


def test_monitor_plots_cpm_object_with_dxo(generate_simu_hdf5):
    data = ExperimentalData("example:simulation_cpm")
    reconstruction = Reconstruction(data, Params())

    monitor = Monitor()
    monitor.reconstruction = reconstruction

    assert_allclose(monitor.objectPixelSize, reconstruction.dxo)
