import logging
from tkinter import Tk, filedialog

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

import PtyLab
from PtyLab import Engines


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


# =============================================================================
# File selection
# =============================================================================

def select_hdf5_file(title):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("HDF5 files", "*.hdf5"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    return file_path


print()
print("=" * 70)
print("Purity-based parameter calibration")
print("=" * 70)

print()
print(
    "Step 1: Select the ptychography dataset to calibrate.\n"
    "This should be the HDF5 file containing the measured diffraction data."
)

filePath = select_hdf5_file(
    "Select ptychography dataset"
)

if not filePath:
    raise RuntimeError(
        "No ptychography dataset was selected."
    )

print()
print("Selected dataset:")
print(filePath)


# =============================================================================
# Optional seed reconstruction
# =============================================================================

print()
print(
    "Step 2: Choose the initial probe.\n"
    "If you already have a previous reconstruction, it can be used as a "
    "seed probe.\n"
    "Otherwise, purityPIE can start from a circular probe estimate."
)

use_seed = input(
    "Use an existing reconstruction as seed probe? [y/N]: "
).strip().lower() in ("y", "yes")


filePath_recon = None

if use_seed:

    print()
    print(
        "Select the HDF5 reconstruction file containing the seed probe."
    )

    filePath_recon = select_hdf5_file(
        "Select seed reconstruction"
    )

    if not filePath_recon:
        raise RuntimeError(
            "Seed probe was requested, but no reconstruction file "
            "was selected."
        )

    print()
    print("Selected seed reconstruction:")
    print(filePath_recon)

else:

    print()
    print(
        "No seed reconstruction selected.\n"
        "The reconstruction will start from a circular probe "
        "and a uniform object."
    )


# =============================================================================
# Load data
# =============================================================================

experimentalData, reconstruction, params, monitor, _ = (
    PtyLab.easyInitialize(
        filePath,
        operationMode="CPM",
    )
)

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

params.purityCalibrationTarget = (
    CALIBRATION_TARGET
)

# Shared settings
params.purityAdaptive = True
params.purityMaxEvaluations = 20
params.purityScanPlot = True

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
    print(
        f"Initial z guess: "
        f"{reconstruction.zo * 1e3:.6f} mm"
    )


elif CALIBRATION_TARGET == "wavelength":
    
    reconstruction.zo = initial_z

    print(
                f"Fixed z used for wavelength calibration: "
                f"{reconstruction.zo * 1e3:.6f} mm"
            )
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
        reconstruction.wavelength
        * reconstruction.zo
        / reconstruction.Ld
    )

    print()
    print(
        f"Initial wavelength guess: "
        f"{reconstruction.wavelength * 1e9:.6f} nm"
    )


else:

    raise ValueError(
        "CALIBRATION_TARGET must be either "
        "'z' or 'wavelength'."
    )


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
    reconstruction.initialProbe_filename = (
        filePath_recon
    )

    print()
    print(
        "Initial probe: loaded from seed reconstruction"
    )

else:

    reconstruction.initialProbe = "circ"

    print()
    print(
        "Initial probe: circular estimate"
    )

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

    print(
        f"Entrance pupil diameter: "
        f"{pupil_diameter * 1e6:.2f} um"
    )


reconstruction.initializeObjectProbe()

reconstruction.describe_reconstruction()


# =============================================================================
# Monitor settings
# =============================================================================

monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"
monitor.verboseLevel = "low"

monitor.objectZoom = None
monitor.probeZoom = None


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

params.comStabilizationSwitch = 10

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
purity_engine.numIterations = 5

best_value, best_purity = (
    purity_engine.reconstruct()
)


# =============================================================================
# Print results
# =============================================================================

print()
print("=" * 70)
print("Purity-based calibration finished")
print("=" * 70)


if CALIBRATION_TARGET == "z":

    print(
        f"Initial z : "
        f"{purity_engine.purityZInitialGuess * 1e3:.6f} mm"
    )

    print(
        f"Best z    : "
        f"{best_value * 1e3:.6f} mm"
    )

    print(
        f"Delta z   : "
        f"{(
            best_value
            - purity_engine.purityZInitialGuess
        ) * 1e6:+.3f} um"
    )

    print(
        f"Best purity: "
        f"{best_purity:.6f}"
    )

    print()
    print("z scan:")

    for z, purity in zip(
        purity_engine.purityZValues,
        purity_engine.purityZMetrics,
    ):

        print(
            f"z = {z * 1e3:.6f} mm | "
            f"delta z = "
            f"{(
                z
                - purity_engine.purityZInitialGuess
            ) * 1e6:+.2f} um | "
            f"purity = {purity:.6f}"
        )


elif CALIBRATION_TARGET == "wavelength":

    print(
        f"Initial wavelength : "
        f"{purity_engine.purityWavelengthInitialGuess * 1e9:.6f} nm"
    )

    print(
        f"Best wavelength    : "
        f"{best_value * 1e9:.6f} nm"
    )

    print(
        f"Delta wavelength   : "
        f"{(
            best_value
            - purity_engine.purityWavelengthInitialGuess
        ) * 1e9:+.4f} nm"
    )

    print(
        f"Best purity        : "
        f"{best_purity:.6f}"
    )

    print()
    print("Wavelength scan:")

    for wavelength, purity in zip(
        purity_engine.purityWavelengthValues,
        purity_engine.purityWavelengthMetrics,
    ):

        print(
            f"wavelength = "
            f"{wavelength * 1e9:.6f} nm | "
            f"delta wavelength = "
            f"{(
                wavelength
                - purity_engine.purityWavelengthInitialGuess
            ) * 1e9:+.4f} nm | "
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

# reconstruction.saveResults(
#     "purity_calibrated_recon.hdf5",
#     squeeze=False,
# )

plt.show(block=True)