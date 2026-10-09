from dataclasses import dataclass
from typing import Any

import numpy as np

from PtyLab.utils.gpuUtils import asNumpyArray
from PtyLab.utils.utils import fft2c

FULL_ROI = (slice(None), slice(None))


def panel_geometry(reconstruction) -> dict:
    """Axis steps, units and labels of the object and probe panels.

    FPM reconstructs the object spectrum and a pupil, so the panels use the FPM
    object sampling `dxo_fpm` and the pupil-plane step `dfp` (a spatial frequency)
    instead of `dxo` and `dxp`.

    Args:
        reconstruction (Reconstruction): Reconstruction whose sampling is shown.

    Returns:
        dict: The `MonitorFrame` fields `object_pixel_size`, `probe_pixel_size`,
            `probe_axis_unit` and `probe_label`.
    """
    if reconstruction.data.operationMode == "FPM":
        return {
            "object_pixel_size": reconstruction.dxo_fpm,
            "probe_pixel_size": reconstruction.dfp,
            "probe_axis_unit": "1/um",
            "probe_label": "Pupil estimate",
        }
    return {
        "object_pixel_size": reconstruction.dxo,
        "probe_pixel_size": reconstruction.dxp,
        "probe_axis_unit": "mm",
        "probe_label": "Probe estimate",
    }


@dataclass
class MonitorFrame:
    """One snapshot of the reconstruction state, handed to a monitor.

    The engine builds one frame per monitor update and passes it to
    `AbstractMonitor.update`. The monitor decides what to show from it.

    The array fields are references to the live reconstruction arrays (CPU or GPU),
    not copies, so building a frame is cheap even when the monitor ignores it. They
    are only valid during the `update` call; a monitor that keeps data for later
    must copy it, e.g. through `object_view` and `probe_view`.

    Attributes:
        error (np.ndarray): Error metric of every iteration so far.
        object (Any): `reconstruction.object`. For FPM this is the object spectrum;
            `object_view` transforms it to real space.
        probe (Any): `reconstruction.probe` (the pupil for FPM).
        operation_mode (str): `"CPM"` or `"FPM"`.
        object_roi (tuple[slice, slice]): Region of the object to display.
        probe_roi (tuple[slice, slice]): Region of the probe to display.
        object_pixel_size (float): Axis step of the object panel in meters.
        probe_pixel_size (float): Axis step of the probe panel, in meters for CPM
            and in 1/m for FPM.
        probe_axis_unit (str): Unit of the probe panel axes.
        probe_label (str): Title of the probe panel.
        nosm (int): Number of object state mixtures.
        npsm (int): Number of probe state mixtures.
        purity_object (Any): Object purity, meaningful only when `nosm > 1`.
        purity_probe (Any): Probe purity, meaningful only when `npsm > 1`.
        zo (Any): Sample-detector distance in meters.
        positions (np.ndarray | None): Scan positions in object pixels.
        engine_name (str | None): Name of the engine that produced the frame.
        encoder_original (np.ndarray | None): Measured encoder positions.
        encoder_corrected (np.ndarray | None): Encoder positions after position
            correction.
        beam_width (tuple[float, float] | None): `(y, x)` probe FWHM in meters.
        I_estimated (np.ndarray | None): Estimated detector intensity at the
            last scan position, only filled for `verboseLevel == "high"`.
        I_measured (np.ndarray | None): Measured detector intensity at the same
            position.
        area_overlap (float | None): Estimated probe area overlap.
        linear_overlap (float | None): Estimated probe linear overlap.
    """

    error: np.ndarray
    object: Any
    probe: Any
    operation_mode: str = "CPM"
    object_roi: tuple = FULL_ROI
    probe_roi: tuple = FULL_ROI
    object_pixel_size: float = 1.0
    probe_pixel_size: float = 1.0
    probe_axis_unit: str = "mm"
    probe_label: str = "Probe estimate"
    nosm: int = 1
    npsm: int = 1
    purity_object: Any = None
    purity_probe: Any = None
    zo: Any = None
    positions: np.ndarray | None = None
    engine_name: str | None = None
    encoder_original: np.ndarray | None = None
    encoder_corrected: np.ndarray | None = None
    beam_width: tuple | None = None
    I_estimated: np.ndarray | None = None
    I_measured: np.ndarray | None = None
    area_overlap: float | None = None
    linear_overlap: float | None = None

    @classmethod
    def from_reconstruction(cls, reconstruction, **fields) -> "MonitorFrame":
        """Fill the fields that follow from the reconstruction state.

        The panel sampling comes from `panel_geometry`. The engine must have
        initialised `reconstruction.error` first.

        Args:
            reconstruction (Reconstruction): Current reconstruction state.
            **fields: Further `MonitorFrame` fields (ROIs, engine name, encoder,
                beam width, diffraction data, overlaps), passed through unchanged.

        Returns:
            MonitorFrame: The snapshot.
        """
        return cls(
            error=reconstruction.error,
            object=reconstruction.object,
            probe=reconstruction.probe,
            operation_mode=reconstruction.data.operationMode,
            **panel_geometry(reconstruction),
            nosm=reconstruction.nosm,
            npsm=reconstruction.npsm,
            purity_object=reconstruction.purityObject,
            purity_probe=reconstruction.purityProbe,
            zo=reconstruction.zo,
            positions=reconstruction.positions,
            **fields,
        )

    @property
    def iteration(self) -> int:
        """Number of completed iterations."""
        return len(self.error)

    def object_view(self) -> np.ndarray:
        """Real-space object inside `object_roi`, as a squeezed NumPy array.

        Leading singleton axes of the 6D array `(nlambda, nosm, 1, nslice, No, No)`
        are dropped, so a single-mode object comes back as `(Ny, Nx)`.
        """
        # FPM stores the object spectrum; the monitor shows its fft2c, as it always has
        obj =fft2c(self.object) if self.operation_mode == "FPM" else self.object
        # crop before moving to the CPU so only the displayed region is copied
        return np.squeeze(asNumpyArray(obj[..., self.object_roi[0], self.object_roi[1]]))

    def probe_view(self) -> np.ndarray:
        """Probe inside `probe_roi`, as a squeezed NumPy array."""
        probe = self.probe[..., self.probe_roi[0], self.probe_roi[1]]
        return np.squeeze(asNumpyArray(probe))
