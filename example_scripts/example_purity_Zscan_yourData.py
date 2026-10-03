import logging
from tkinter import Tk, filedialog

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np

import PtyLab
from PtyLab import Engines


logging.basicConfig(level=logging.INFO)


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

    if not file_path:
        raise RuntimeError(f"No file selected: {title}")

    return file_path


# -------------------------------------------------------------------------
# Select input files
# -------------------------------------------------------------------------

filePath = select_hdf5_file(
    "Select ptychography dataset"
)

filePath_recon = select_hdf5_file(
    "Select seed reconstruction"
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


initial_z = reconstruction.zo 

print(f"Initial z guess: {reconstruction.zo * 1e3:.6f} mm")


# -------------------------------------------------------------------------
# Reconstruction dimensions
# -------------------------------------------------------------------------

# Purity-based calibration requires multiple probe modes.
reconstruction.npsm = 4

reconstruction.nosm = 1
reconstruction.nlambda = 1
reconstruction.nslice = 1


# -------------------------------------------------------------------------
# Object and probe initialization
# -------------------------------------------------------------------------

reconstruction.initialProbe = "recon"
reconstruction.initialProbe_filename = filePath_recon
reconstruction.initialObject = "ones"

reconstruction.initializeObjectProbe()

reconstruction.describe_reconstruction()


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

params.purityZScanSwitch = True

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

# Plot purity versus z after the scan.
params.purityZScanPlot = True


monitor.describe_parameters(params)


# -------------------------------------------------------------------------
# Run purity-based axial calibration
# -------------------------------------------------------------------------

if params.purityZScanSwitch:

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