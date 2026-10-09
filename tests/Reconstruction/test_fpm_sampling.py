import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.frame import panel_geometry
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.visualisation import plotExtent


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
    panels = panel_geometry(fpm_reconstruction)

    assert_allclose(panels["object_pixel_size"], fpm_reconstruction.dxo_fpm)
    # the plotted extent is the field of view of the raw images, not No * dxo
    assert_allclose(
        fpm_reconstruction.No * panels["object_pixel_size"],
        fpm_reconstruction.Np * fpm_reconstruction.dxp,
    )


def test_monitor_plots_cpm_object_with_dxo(generate_simu_hdf5):
    data = ExperimentalData("example:simulation_cpm")
    reconstruction = Reconstruction(data, Params())

    panels = panel_geometry(reconstruction)

    assert_allclose(panels["object_pixel_size"], reconstruction.dxo)


def test_fpm_pupil_sampling_matches_the_numerical_aperture(fpm_reconstruction):
    """The pupil grid step is what puts the NA cut-off where the code puts it."""
    reconstruction = fpm_reconstruction
    assert_allclose(reconstruction.dfp, 1 / (reconstruction.Np * reconstruction.dxp))

    # radius of the aperture, in pixels, derived two independent ways
    assert_allclose(
        (reconstruction.NA / reconstruction.wavelength) / reconstruction.dfp,
        (reconstruction.data.entrancePupilDiameter / 2) / reconstruction.dxp,
    )


def test_monitor_plots_fpm_pupil_in_reciprocal_units(fpm_reconstruction):
    """For FPM the probe panel shows the pupil, which lives in Fourier space."""
    panels = panel_geometry(fpm_reconstruction)

    assert panels["probe_label"] == "Pupil estimate"
    assert panels["probe_axis_unit"] == "1/um"
    assert_allclose(panels["probe_pixel_size"], fpm_reconstruction.dfp)

    # the axis spans the bandwidth the low-resolution grid can carry, 1 / dxp
    assert_allclose(
        fpm_reconstruction.Np * panels["probe_pixel_size"], 1 / fpm_reconstruction.dxp
    )


def test_monitor_probe_panel_unchanged_for_cpm(generate_simu_hdf5):
    """CPM reconstructs a real-space probe, so that panel must not move."""
    data = ExperimentalData("example:simulation_cpm")
    reconstruction = Reconstruction(data, Params())

    panels = panel_geometry(reconstruction)

    assert panels["probe_label"] == "Probe estimate"
    assert panels["probe_axis_unit"] == "mm"
    assert_allclose(panels["probe_pixel_size"], reconstruction.dxp)


def test_only_reciprocal_axes_are_centred_on_zero(generate_simu_hdf5):
    """Real-space extents keep the historical corner origin."""
    shape = (8, 8)

    assert plotExtent(2e-6, "mm", shape) == [0, 2e-6 * 8 * 1e3, 2e-6 * 8 * 1e3, 0]

    left, right, bottom, top = plotExtent(3e3, "1/um", shape)
    assert left == -right and top == -bottom
    assert_allclose(right, 3e3 * 8 * 1e-6 / 2)
