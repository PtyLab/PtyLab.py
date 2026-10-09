"""
Live reconstruction monitor, in GUI windows or in Jupyter cells. To mainly explore the
monitor visualization options

Runs a short mPIE reconstruction and shows the default `Monitor`: object, probe and
error metric; the estimated vs measured diffraction intensity; and the history of
the other quantities the engine changes (zo, purities, scan positions).

As a script, the monitor opens its own windows, which stay open after the run
until you close them:

    $ python example_scripts/exampleMonitorGUI.py [--file <filename>] [--iterations 20]
          [--autofocus] [--position-correction] [--modes 3]

Cell by cell (`# %%`, e.g. "Run Cell" in VS Code), the figures appear as cell
outputs and update in place. Change the settings by editing `args` in the
settings cell; re-running the reconstruction cell continues from the current
estimate.

The flags switch on reconstruction features that change those quantities, so
their panels appear in the history window.
"""

# %% Imports
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import PtyLab
from PtyLab import Engines
from PtyLab.io import getExampleDataFolder
from PtyLab.Monitor.Plots import is_inline

# %% Settings
# allow_abbrev=False: the Jupyter kernel's own `--f=<kernel>.json` must not be
# read as an abbreviation of `--file`
parser = argparse.ArgumentParser(description="Live monitor preview", allow_abbrev=False)
parser.add_argument("--file", type=str, default=f"{getExampleDataFolder()}/simu.hdf5")
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--autofocus", action="store_true", help="TV autofocus: zo changes")
parser.add_argument(
    "--position-correction",
    action="store_true",
    help="position correction: scan positions change",
)
parser.add_argument(
    "--modes",
    type=int,
    default=1,
    help="number of probe modes; >1 with orthogonalization changes the purity",
)
# parse_known_args ignores the arguments a Jupyter kernel is started with
args, _ = parser.parse_known_args()

# when running cell by cell, override here, e.g.
# args.autofocus = True
# args.modes = 3

# a script needs a GUI backend to open windows; matplotlib falls back to a
# non-interactive one (e.g. Agg) when there is no display
backend = plt.get_backend().lower()
if not is_inline() and backend in ("agg", "pdf", "ps", "svg", "cairo", "template"):
    sys.exit(
        f"matplotlib backend is '{backend}', which cannot open a window. "
        "Run this on a machine with a display, or set MPLBACKEND=QtAgg / TkAgg."
    )

# %% Data and parameters
experimentalData, reconstruction, params, monitor, _ = PtyLab.easyInitialize(
    args.file, operationMode="CPM"
)
params.gpuSwitch = False
params.comStabilizationSwitch = True

if args.autofocus:
    params.TV_autofocus = True
    params.TV_autofocus_run_every = 1  # update zo every iteration
if args.position_correction:
    params.positionCorrectionSwitch = True
if args.modes > 1:
    reconstruction.npsm = args.modes
    # the purity is only recomputed when the modes are orthogonalized
    params.orthogonalizationSwitch = True
    params.orthogonalizationFrequency = 1

monitor.figureUpdateFrequency = 1  # redraw after every iteration
monitor.verboseLevel = "high"  # also show the diffraction intensities
monitor.objectPlot = "complex"  # complex | abs | angle
monitor.showParameterHistory = True  # third figure: zo, purity, scan positions

reconstruction.initializeObjectProbe()

# mPIE calls showReconstruction every iteration (ePIE currently does not)
engine = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine.numIterations = args.iterations
if args.position_correction:
    engine.startAtIteration = 1  # position correction is off before this iteration

# %% Reconstruct (re-run to continue from the current estimate)
engine.reconstruct()

# %% Save the reconstruction plus a PNG of every monitor figure
# results/monitor.hdf5, results/monitor_reconstruction.png, ...
Path("results").mkdir(exist_ok=True)
reconstruction.saveResults("results/monitor.hdf5", snapshot=monitor)

# %% Keep the windows open (script only)
if not is_inline():
    # the monitor runs in interactive mode; switch it off so show() blocks and
    # the windows stay open after the reconstruction finishes
    plt.ioff()
    plt.show()
