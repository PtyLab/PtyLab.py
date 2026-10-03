import logging
from tkinter import Tk, filedialog

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np

import PtyLab
from PtyLab import Engines


logging.basicConfig(level=logging.INFO)


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
print("Purity-based axial calibration")
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

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

experimentalData, reconstruction, params, monitor, _ = (
    PtyLab.easyInitialize(
        filePath,
        operationMode="CPM",
    )
)

#experimentalData.setOrientation(0)

print("Loaded file:", experimentalData.filename)
print("Ptychogram shape:", experimentalData.ptychogram.shape)
print("Initial z:", reconstruction.zo)
print("Wavelength:", reconstruction.wavelength)


# Use the z value stored in the dataset by default.
initial_z = reconstruction.zo

# Uncomment to set the initial z guess manually.
# initial_z = 32.5e-3

experimentalData.zo = initial_z
reconstruction.zo = initial_z

print(f"Initial z guess: {initial_z * 1e3:.6f} mm")


# -------------------------------------------------------------------------
# Reconstruction dimensions
# -------------------------------------------------------------------------

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

    # Circular initialization requires an entrance pupil diameter.
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


# -------------------------------------------------------------------------
# Monitor settings
# -------------------------------------------------------------------------

monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"
monitor.verboseLevel = "low"

monitor.objectZoom = None
monitor.probeZoom = None


# -------------------------------------------------------------------------
# General reconstruction parameters
# -------------------------------------------------------------------------

params.positionOrder = "random"

# Start with the propagator used for your actual dataset.
params.propagatorType = "Fraunhofer"

params.gpuSwitch = True

params.intensityConstraint = "standard"

params.probePowerCorrectionSwitch = True

params.probeSmoothenessSwitch = True
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10
params.comStabilizationSwitch = 10
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10
params.couplingSwitch = True
params.couplingAleph = 1


# -------------------------------------------------------------------------
# Purity-based z calibration parameters
# -------------------------------------------------------------------------
# Half-range around the current reconstruction.zo.
#
# Example:
# reconstruction.zo = 50 mm
# purityZScanRange = 100 µm
#
# scan range:
# 49.9 mm ... 50.1 mm
params.purityZScanRange = 2e-3

# Number of candidate z values.
params.purityZScanPoints = 10

params.purityZAdaptive = True

params.purityZInitialStep = 200e-6
params.purityZStepGrowth = 1.5
params.purityZStepShrink = 0.5

params.purityZMinStep = 20e-6
params.purityZPurityTolerance = 1e-4
params.purityZMaxEvaluations = 20

# Plot purity versus z after the scan.
params.purityZScanPlot = True
monitor.describe_parameters(params)


# -------------------------------------------------------------------------
# Run purity-based axial calibration
# -------------------------------------------------------------------------



purity_engine = Engines.purityPIE(
    reconstruction,
    experimentalData,
    params,
    monitor,
)

# Number of mPIE iterations used to evaluate EACH candidate z.
purity_engine.numIterations = 5

best_z, best_purity = purity_engine.reconstruct()

print()
print("Purity-based axial calibration finished")
print("---------------------------------------")
print(f"Initial z : {purity_engine.purityZInitialGuess * 1e3:.6f} mm")
print(f"Best z    : {best_z * 1e3:.6f} mm")
print(
    f"Delta z   : "
    f"{(best_z - purity_engine.purityZInitialGuess) * 1e6:.3f} um"
)
print(f"Best purity: {best_purity:.6f}")

print()
print("z scan:")
for z, purity in zip(
    purity_engine.purityZValues,
    purity_engine.purityZMetrics,
):
    print(
        f"z = {z * 1e3:.6f} mm, "
        f"delta z = {(z - purity_engine.purityZInitialGuess) * 1e6:+.2f} um, "
        f"purity = {purity:.6f}"
    )



# -------------------------------------------------------------------------
# Optional: continue reconstruction using the best-z state
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Save / display
# -------------------------------------------------------------------------

# reconstruction.saveResults(
#     f"{getExampleDataFolder()}/purity_calibrated_recon.hdf5",
#     squeeze=False,
# )

plt.show(block=True)