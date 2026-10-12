import logging
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

import PtyLab
from PtyLab import Engines
from PtyLab.io.generateSimulationData import generate_simu_hdf5

logging.basicConfig(level=logging.INFO)


# =============================================================================
# User settings
# =============================================================================

# Choose which experimental parameter to calibrate:
#
# "z"          -> sample-to-detector distance
# "wavelength" -> illumination wavelength
CALIBRATION_TARGET = "z"  # "z" or "wavelength"

# Z / Wavelength calibration assumes that the wavelength / Z is known to a resonable degree of accuracy."
initial_z = 35e-3  # 32.5 mm

initial_wavelength = 13.5e-9  # 13.5 nm

# HDF5 file with the measured diffraction data. Leave empty ("") to be asked
# for the path when the script runs.
DATASET_PATH = ""

# Optional previous reconstruction whose probe is used as the starting guess.
# None -> start from a circular probe; "" -> asked when the script runs.
SEED_RECON_PATH = None

# Run on a simulated dataset instead of DATASET_PATH, e.g. to try the script
# or watch the monitor. The data is generated in a temporary folder, and the
# calibration starts from the simulated z / wavelength shifted by the offsets
# below (initial_z and initial_wavelength above are then ignored).
# On this simulation the purity hardly changes with z, so the calibrated value
# is not meaningful; use it to check that everything runs.
USE_SYNTHETIC_DATA = True
SYNTHETIC_Z_OFFSET = 1e-3  # 1 mm, inside purityZScanRange
SYNTHETIC_WAVELENGTH_OFFSET = 0.5e-9  # 0.5 nm, inside purityWavelengthScanRange


# =============================================================================
# File selection
# =============================================================================


# Datasets PtyLab needs in a CPM file (see ExperimentalData._setFields) and the
# dataset read from a seed reconstruction (see Reconstruction.loadResults).
DATASET_FIELDS = ["ptychogram", "wavelength", "encoder", "dxd", "zo"]
SEED_FIELDS = ["probe"]


def check_hdf5_file(path, required_fields, description):
    """Return `path` as a Path after checking it is an HDF5 file holding `required_fields`.

    Catches a wrong file before the scan starts instead of minutes into it.
    """
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    if not h5py.is_hdf5(path):
        raise ValueError(f"{description} is not an HDF5 file: {path}")

    with h5py.File(path, "r") as archive:
        missing = [key for key in required_fields if key not in archive]

    if missing:
        raise KeyError(
            f"{description} {path} is missing the dataset(s) {missing}. "
            f"Expected: {required_fields}"
        )

    return path


def ask_path(prompt):
    # strip quotes so a path pasted from a file manager also works
    return input(prompt).strip().strip("'\"")


print()
print("=" * 70)
print("Purity-based parameter calibration")
print("=" * 70)

print()
print(
    "Step 1: The ptychography dataset to calibrate.\n"
    "This should be the HDF5 file containing the measured diffraction data."
)

if USE_SYNTHETIC_DATA:
    DATASET_PATH = Path(tempfile.mkdtemp(prefix="ptylab_purity_")) / "simu.hdf5"
    print(f"Generating synthetic data in {DATASET_PATH} ...")
    generate_simu_hdf5(DATASET_PATH)

if not DATASET_PATH:
    DATASET_PATH = ask_path("Path to the ptychography dataset (.hdf5): ")

if not DATASET_PATH:
    raise RuntimeError("No ptychography dataset was given.")

filePath = check_hdf5_file(DATASET_PATH, DATASET_FIELDS, "Ptychography dataset")

print()
print("Dataset:")
print(filePath)


# =============================================================================
# Optional seed reconstruction
# =============================================================================

print()
print(
    "Step 2: The initial probe.\n"
    "If you already have a previous reconstruction, it can be used as a "
    "seed probe.\n"
    "Otherwise, purityPIE can start from a circular probe estimate."
)

if SEED_RECON_PATH == "":
    SEED_RECON_PATH = (
        ask_path(
            "Path to a reconstruction to use as seed probe "
            "(leave empty for a circular probe): "
        )
        or None
    )

filePath_recon = None

if SEED_RECON_PATH is not None:
    filePath_recon = check_hdf5_file(
        SEED_RECON_PATH, SEED_FIELDS, "Seed reconstruction"
    )

    print()
    print("Seed reconstruction:")
    print(filePath_recon)

else:
    print()
    print(
        "No seed reconstruction given.\n"
        "The reconstruction will start from a circular probe "
        "and a uniform object."
    )


# =============================================================================
# Load data
# =============================================================================

experimentalData, reconstruction, params, monitor, _ = PtyLab.easyInitialize(
    filePath,
    operationMode="CPM",
)

if USE_SYNTHETIC_DATA:
    # the simulation is stored in the orientation PtyLab expects; start the
    # calibration a known distance away from the simulated values
    initial_z = reconstruction.zo + SYNTHETIC_Z_OFFSET
    initial_wavelength = reconstruction.wavelength + SYNTHETIC_WAVELENGTH_OFFSET
else:
    # Change only if required by your dataset.
    experimentalData.setOrientation(4)

print()
print("Loaded file:", experimentalData.filename)
print("Ptychogram shape:", experimentalData.ptychogram.shape)
print(f"Dataset z: {reconstruction.zo * 1e3:.6f} mm")
print(f"Dataset wavelength: {reconstruction.wavelength * 1e9:.6f} nm")


# =============================================================================
# Purity-based calibration settings
# =============================================================================

params.purityCalibrationTarget = CALIBRATION_TARGET

# Shared settings
params.purityAdaptive = True
params.purityMaxEvaluations = 20

# -----------------------------------------------------------------------------
# z calibration settings
# -----------------------------------------------------------------------------

# Used by the fixed-grid scan.
params.purityZScanRange = 2e-3
params.purityZScanPoints = 10

# Used by the adaptive scan.
params.purityZInitialStep = 200e-6
params.purityZStepGrowth = 1.5
params.purityZStepShrink = 0.5
params.purityZMinStep = 20e-6
params.purityZPurityTolerance = 1e-4


# -----------------------------------------------------------------------------
# Wavelength calibration settings
# -----------------------------------------------------------------------------

# Used by the fixed-grid scan.
params.purityWavelengthScanRange = 1.0e-9
params.purityWavelengthScanPoints = 11

# Used by the adaptive scan.
params.purityWavelengthInitialStep = 0.1e-9
params.purityWavelengthStepGrowth = 1.5
params.purityWavelengthStepShrink = 0.5
params.purityWavelengthMinStep = 0.01e-9
params.purityWavelengthPurityTolerance = 1e-4

# =============================================================================
# Initial calibration guess
# =============================================================================

if CALIBRATION_TARGET == "z":
    # Use the value stored in the dataset by default.
    # initial_z = reconstruction.zo

    # Uncomment to set the initial z guess manually.
    # initial_z = 30e-3

    experimentalData.zo = initial_z
    reconstruction.zo = initial_z

    print()
    print(f"Initial z guess: {reconstruction.zo * 1e3:.6f} mm")


elif CALIBRATION_TARGET == "wavelength":
    reconstruction.zo = initial_z

    print(f"Fixed z used for wavelength calibration: {reconstruction.zo * 1e3:.6f} mm")
    # Use the wavelength stored in the dataset by default.
    # initial_wavelength = reconstruction.wavelength

    # Uncomment to set the initial wavelength manually.
    # initial_wavelength = 13.0e-9

    experimentalData.wavelength = initial_wavelength
    reconstruction.wavelength = initial_wavelength

    # Keep the single-wavelength spectral-density entry consistent.
    if reconstruction.nlambda == 1:
        experimentalData.spectralDensity = [initial_wavelength]
        reconstruction.spectralDensity = [initial_wavelength]

    # Update CPM sampling because wavelength has no property setter.
    reconstruction.dxp = (
        reconstruction.wavelength * reconstruction.zo / reconstruction.Ld
    )

    print()
    print(f"Initial wavelength guess: {reconstruction.wavelength * 1e9:.6f} nm")


else:
    raise ValueError("CALIBRATION_TARGET must be either 'z' or 'wavelength'.")


# =============================================================================
# Reconstruction dimensions
# =============================================================================

# Purity-based calibration requires multiple probe modes.
reconstruction.npsm = 4

reconstruction.nosm = 1
reconstruction.nlambda = 1
reconstruction.nslice = 1


# =============================================================================
# Object and probe initialization
# =============================================================================

reconstruction.initialObject = "ones"

if filePath_recon is not None:
    reconstruction.initialProbe = "recon"
    reconstruction.initialProbe_filename = filePath_recon

    print()
    print("Initial probe: loaded from seed reconstruction")

else:
    reconstruction.initialProbe = "circ"

    print()
    print("Initial probe: circular estimate")

    pupil_diameter = getattr(
        experimentalData,
        "entrancePupilDiameter",
        None,
    )

    if pupil_diameter is None:
        raise RuntimeError(
            "Circular probe initialization requires "
            "`experimentalData.entrancePupilDiameter`.\n"
            "Define the probe / entrance pupil diameter before calling "
            "`initializeObjectProbe()`, or provide a seed reconstruction."
        )

    print(f"Entrance pupil diameter: {pupil_diameter * 1e6:.2f} um")


reconstruction.initializeObjectProbe()

reconstruction.describe_reconstruction()


# =============================================================================
# Monitor settings
# =============================================================================

monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"
monitor.verboseLevel = "low"

# default settings as no zoom
monitor.objectZoom = 1
monitor.probeZoom = 1


# =============================================================================
# General reconstruction parameters
# =============================================================================

params.positionOrder = "random"

# Use the propagator appropriate for your dataset.
params.propagatorType = "Fraunhofer"

params.gpuSwitch = True

params.intensityConstraint = "standard"

params.positionCorrectionSwitch = False

params.modulusEnforcedProbeSwitch = False
params.probePowerCorrectionSwitch = True

params.probeSmoothenessSwitch = True
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10

params.comStabilizationSwitch = True

params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10

params.absorbingProbeBoundary = False
params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False

params.couplingSwitch = True
params.couplingAleph = 1


monitor.describe_parameters(params)


# =============================================================================
# Run purity-based calibration
# =============================================================================

purity_engine = Engines.purityPIE(
    reconstruction,
    experimentalData,
    params,
    monitor,
)

# Number of mPIE iterations used to evaluate EACH candidate.
purity_engine.numIterations = 50

best_value, best_purity = purity_engine.reconstruct()


# =============================================================================
# Print results
# =============================================================================

print()
print("=" * 70)
print("Purity-based calibration finished")
print("=" * 70)


if CALIBRATION_TARGET == "z":
    print(f"Initial z : {purity_engine.purityZInitialGuess * 1e3:.6f} mm")

    print(f"Best z    : {best_value * 1e3:.6f} mm")

    print(
        f"Delta z   : {(best_value - purity_engine.purityZInitialGuess) * 1e6:+.3f} um"
    )

    print(f"Best purity: {best_purity:.6f}")

    print()
    print("z scan:")

    for z, purity in zip(
        purity_engine.purityZValues,
        purity_engine.purityZMetrics,
    ):
        print(
            f"z = {z * 1e3:.6f} mm | "
            f"delta z = "
            f"{(z - purity_engine.purityZInitialGuess) * 1e6:+.2f} um | "
            f"purity = {purity:.6f}"
        )


elif CALIBRATION_TARGET == "wavelength":
    print(
        f"Initial wavelength : "
        f"{purity_engine.purityWavelengthInitialGuess * 1e9:.6f} nm"
    )

    print(f"Best wavelength    : {best_value * 1e9:.6f} nm")

    print(
        f"Delta wavelength   : "
        f"{(best_value - purity_engine.purityWavelengthInitialGuess) * 1e9:+.4f} nm"
    )

    print(f"Best purity        : {best_purity:.6f}")

    print()
    print("Wavelength scan:")

    for wavelength, purity in zip(
        purity_engine.purityWavelengthValues,
        purity_engine.purityWavelengthMetrics,
    ):
        # computed outside the f-string: line breaks inside {...} need Python 3.12+
        delta_wavelength = wavelength - purity_engine.purityWavelengthInitialGuess
        print(
            f"wavelength = "
            f"{wavelength * 1e9:.6f} nm | "
            f"delta wavelength = "
            f"{delta_wavelength * 1e9:+.4f} nm | "
            f"purity = {purity:.6f}"
        )


# =============================================================================
# Optional: continue reconstruction using the calibrated state
# =============================================================================

continue_with_mPIE = False

if continue_with_mPIE:
    mPIE_engine = Engines.mPIE(
        reconstruction,
        experimentalData,
        params,
        monitor,
    )

    mPIE_engine.numIterations = 50

    mPIE_engine.reconstruct()


# =============================================================================
# Save / display
# =============================================================================

reconstruction.saveResults(
    "purity_calibrated_recon.hdf5",
    squeeze=False,
    snapshot=monitor,  # stores as png
)

plt.show(block=True)
