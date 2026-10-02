import argparse
import logging

import matplotlib

matplotlib.use("qt5agg")

import matplotlib.pyplot as plt
import numpy as np

import PtyLab
from PtyLab import Engines
from PtyLab.io import getExampleDataFolder


logging.basicConfig(level=logging.INFO)


"""
Purity-based axial calibration example.

Workflow
--------
1. Load a conventional ptychography dataset.
2. Initialize a multimode reconstruction.
3. Run purityPIE over a fixed z range.
4. Select the z value that gives the highest reconstructed probe purity.
5. Optionally continue the reconstruction with mPIE from the best-z state.
"""


# -------------------------------------------------------------------------
# Command-line arguments
# -------------------------------------------------------------------------

import logging
import os

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

import PtyLab
from PtyLab import Engines


logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------------
# Data path
# -------------------------------------------------------------------------

PATH = r"\\dionysios.iap.uni-jena.de\faserlaser$\AG_Imaging/2023_XUV_Bio-imaging\XUV imaging\_RAWDATA"
subdir = "20240522/176"

filePath = os.path.join(
    PATH,
    f"{subdir}/dp/dp_processed.hdf5",
)

filePath_recon = os.path.join(
    PATH,
    f"{subdir}/recons/seed.hdf5",
)


# -------------------------------------------------------------------------
# Experimental parameters
# -------------------------------------------------------------------------

initial_z = 32.0e-3
wavelength = 13.5e-9
bin_factor = 1

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

experimentalData, reconstruction, params, monitor, _ = (
    PtyLab.easyInitialize(
        filePath,
        operationMode="CPM",
    )
)

experimentalData.setOrientation(4)

print("Loaded file:", experimentalData.filename)
print("Ptychogram shape:", experimentalData.ptychogram.shape)
print("Initial z:", reconstruction.zo)
print("Wavelength:", reconstruction.wavelength)

# -------------------------------------------------------------------------
# Optional initial z offset for testing
# -------------------------------------------------------------------------

# Uncomment this to deliberately start from an incorrect distance.
#
# experimentalData.zo += 100e-6
# reconstruction.zo += 100e-6



experimentalData.zo = initial_z
reconstruction.zo = initial_z

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


# Optional quadratic phase initialization, if required for your dataset.
#
# reconstruction.probe *= np.exp(
#     1.0j
#     * 2
#     * np.pi
#     / (reconstruction.wavelength * reconstruction.zo * 2)
#     * (reconstruction.Xp**2 + reconstruction.Yp**2)
#     / 2
# )


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
params.fftshiftSwitch = False

params.intensityConstraint = "standard"

params.positionCorrectionSwitch = False

params.modulusEnforcedProbeSwitch = False
params.probePowerCorrectionSwitch = True

params.probeSmoothenessSwitch = True
params.probeSmoothnessAleph = 1e-2
params.probeSmoothenessWidth = 10

params.comStabilizationSwitch = 10

# purityPIE performs an explicit orthogonalization after each candidate-z
# reconstruction. Automatic orthogonalization during the internal
# reconstruction is temporarily disabled by purityZScan().
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10

params.absorbingProbeBoundary = False
params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False

params.couplingSwitch = True
params.couplingAleph = 1

params.TV_autofocus = False
params.l2reg = False


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