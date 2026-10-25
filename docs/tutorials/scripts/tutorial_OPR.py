# %% [markdown]
# # CPM Orthogonal Probe Relaxation
#
# This tutorial reconstructs measured extreme-ultraviolet (EUV) data of a filamentous fungus, recorded while the illumination drifted. Mixed states model the partial coherence within each frame, but they still assume the same probe at every scan position. Orthogonal probe relaxation (OPR) lets the probe change from position to position, within a small set of shared basis probes.
#
# **What you'll learn:**
#
# - How OPR models a probe that varies across the scan
# - How to set the geometry when a file stores placeholder values
# - How to start with mPIE and continue with the `OPR` engine
#
# !!! quote "References"
#     - Orthogonal probe relaxation: M. Odstrčil et al., "Ptychographic coherent diffractive imaging with orthogonal probe relaxation," Opt. Express 24, 8360 (2016). [doi:10.1364/OE.24.008360](https://doi.org/10.1364/OE.24.008360)
#     - Dataset: C. Liu et al., "Visualizing the ultra-structure of microorganisms using table-top extreme ultraviolet imaging," PhotoniX 4, 6 (2023). [doi:10.1186/s43074-023-00084-6](https://doi.org/10.1186/s43074-023-00084-6)
#
# !!! note "Dataset"
#     The dataset (`OPR_dp_processed.hdf5`, about 180 MB) is downloaded automatically into the `example_data/` folder at the project root the first time you run the tutorial. If the file already exists it is not re-downloaded.
#
# !!! warning "GPU required"
#     The `OPR` engine currently runs only on a GPU with CuPy installed; see [Installation](../getting-started/installation.md).
#
# !!! tip "Prerequisites"
#     Complete [CPM Mixed States](tutorial_mixedStates.md) before this tutorial.
#
# !!! info "Run time"
#     The reconstruction takes about a minute on a GPU; most of it is OPR, which keeps and updates a probe for each of the 707 scan positions.

# %% [markdown]
# ## OPR model
# OPR, following M. Odstrčil et al., Opt. Express 24, 8360 (2016), keeps one probe $P_j$ for every scan position $j$. Left free, these probes would have far too many degrees of freedom, so OPR restricts them to a low-dimensional subspace: each $P_j$ is a combination of $K$ shared basis probes $\Phi_k$,
#
# $$
# P_j(\mathbf{r}) = \sum_{k=1}^{K} c_{k,j}\, \Phi_k(\mathbf{r})
# $$
#
# - $\Phi_k$: basis probes, shared by all positions
# - $c_{k,j}$: weight of basis probe $k$ at position $j$
# - $K$: subspace size, `params.OPR_subspace`
#
# After each iteration, OPR stacks the flattened probes as the columns of a matrix $A = [P_1, \dots, P_N]$ and replaces it by its best rank-$K$ approximation from a truncated singular-value decomposition, $A \approx U_K S_K V_K^\dagger$. The new probes blend the current ones with this approximation:
#
# $$
# P_j \leftarrow \alpha\, P_j + (1 - \alpha)\, P_j^{\text{rank } K}
# $$
#
# with $\alpha$ = `params.OPR_alpha` (default 0.05): a larger $\alpha$ lets the probes deviate more from the subspace. OPR combines with mixed states: `params.OPR_modes` selects which of the incoherent probe modes vary with position, and the rest stay the same at every position.

# %% [markdown]
# ## Setup
# Import the libraries and download the dataset.

# %%
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

import PtyLab
from PtyLab import Engines
from PtyLab.io import getExampleDataFolder
from PtyLab.utils.gpuUtils import asNumpyArray

fileName = "OPR_dp_processed.hdf5"
filePath = getExampleDataFolder() / fileName

if not filePath.exists():
    print(f"Downloading {fileName}..")
    urllib.request.urlretrieve("https://ndownloader.figshare.com/files/38461082", filePath)
    print("Download complete.")
else:
    print(f"{fileName} found, skipping download.")

# %% [markdown]
# This file stores placeholder values for the wavelength and detector pixel size, so set the geometry explicitly and call `_setData()` to recompute the derived quantities, such as the detector coordinates. `setOrientation(1)` flips the detector columns to match the direction of the scan axes (see [CPM Mixed States](tutorial_mixedStates.md) for the codes).

# %%
experimentalData = PtyLab.ExperimentalData(filePath, operationMode="CPM")
experimentalData.setOrientation(1)
experimentalData.zo = 31.0e-3  # sample-detector distance
experimentalData.dxd = 2 * 13.5e-6  # detector pixel size, 13.5 um pixels binned 2 x 2
experimentalData.wavelength = 13.5e-9
experimentalData._setData()

print(f"{experimentalData.numFrames} frames of {experimentalData.Nd} x {experimentalData.Nd} pixels")

plt.imshow(np.log10(experimentalData.ptychogram[0] + 1), cmap="gray")
plt.title("First diffraction pattern, log10(I + 1)")
plt.colorbar()
plt.show()

# %% [markdown]
# ## Reconstruction settings
# Use six probe modes for the partial coherence of the EUV source, and enlarge the object array to `No = 2000` pixels so it holds the full scan.

# %%
params = PtyLab.Params()
reconstruction = PtyLab.Reconstruction(experimentalData, params)
reconstruction.No = 2000  # object array size, in pixels
reconstruction.npsm = 6  # number of probe modes
reconstruction.nosm = 1
reconstruction.nlambda = 1
reconstruction.nslice = 1

reconstruction.initialProbe = "circ"
experimentalData.entrancePupilDiameter = 10e-6
reconstruction.initialObject = "ones"
reconstruction.initializeObjectProbe()

monitor = PtyLab.Monitor()
monitor.figureUpdateFrequency = 25
monitor.objectPlot = "complex"
monitor.objectZoom = 2
monitor.probeZoom = 0.1

# %% [markdown]
# Besides the settings from earlier tutorials, two constraints help here:
#
# | Setting | Effect |
# |---------|--------|
# | `absorbingProbeBoundary` | damps the probe towards the edge of its frame, so it cannot wrap around |
# | `absObjectSwitch` | pulls the object towards its amplitude by a fraction `absObjectBeta` each iteration, which suits a sample that mainly absorbs |
#
# The OPR settings: the three strongest probe modes vary with position (`OPR_modes`), within a subspace of five basis probes (`OPR_subspace`). `OPR_tsvd_type = "randomized"` computes the truncated SVD with a faster randomized algorithm.

# %%
params.positionOrder = "random"
params.propagatorType = "Fraunhofer"
params.intensityConstraint = "standard"
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10
params.absorbingProbeBoundary = True
params.absObjectSwitch = True
params.absObjectBeta = 1e-2

# OPR settings
params.OPR_modes = np.array([0, 1, 2])  # probe modes that vary with position
params.OPR_subspace = 5  # number of basis probes K
params.OPR_tsvd_type = "randomized"

# %% [markdown]
# ## Mixed states with mPIE
# Start with 25 iterations of mPIE. It finds a good probe and object to start from, with the same probe at every position. Keep a copy of the object for the comparison at the end.

# %%
engine_mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
engine_mPIE.numIterations = 25
engine_mPIE.betaProbe = 0.25
engine_mPIE.betaObject = 0.25
engine_mPIE.reconstruct()

object_mPIE = reconstruction.object.copy()

# %% [markdown]
# ## OPR
# Continue from these estimates with OPR. Creating the engine copies the modes in `OPR_modes` to every scan position in `reconstruction.probe_stack`.

# %%
engine_OPR = Engines.OPR(reconstruction, experimentalData, params, monitor)
engine_OPR.numIterations = 100
engine_OPR.betaProbe = 0.99
engine_OPR.betaObject = 0.99
engine_OPR.reconstruct()

# %% [markdown]
# ## Comparison
# The probe stack has shape `(nlambda, nosm, len(OPR_modes), nslice, Np, Np, numFrames)`: one probe per mode and scan position. The power of the first mode at each position shows how the illumination changed during the measurement, in the order the frames were recorded: it fluctuates by 10–20 % from frame to frame and drops towards the end of the scan.

# %%
probe_stack = asNumpyArray(reconstruction.probe_stack)  # stays on the GPU after OPR
power = np.sum(np.abs(probe_stack[0, 0, 0, 0]) ** 2, axis=(0, 1))  # (numFrames,)

plt.plot(power / power.mean())
plt.xlabel("frame index")
plt.ylabel("relative power of mode 1")
plt.title("Probe power across the scan")
plt.show()

# %% [markdown]
# The probes at the first and the last scan position, for the dominant mode. Their difference is the drift that a single shared probe cannot represent.

# %%
from PtyLab.utils.visualisation import hsvplot

first, last = probe_stack[0, 0, 0, 0, ..., 0], probe_stack[0, 0, 0, 0, ..., -1]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, P, title in zip(axes, [first, last], ["first position", "last position"]):
    hsvplot(P, ax=ax, pixelSize=reconstruction.dxp, axisUnit="um")
    ax.set_title(f"Mode 1, {title}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Compare the object amplitude after mPIE and after OPR, cropped to the scanned area. With a probe that follows the drift, OPR removes most of the grainy artifacts in the mPIE result.

# %%
crop = (slice(400, 1600), slice(400, 1600))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, obj, title in zip(axes, [object_mPIE, reconstruction.object], ["mPIE", "mPIE + OPR"]):
    amplitude = np.abs(obj[0, 0, 0, 0][crop])
    ax.imshow(amplitude, cmap="gray", vmax=np.percentile(amplitude, 99.9))
    ax.set_title(f"Object amplitude, {title}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# Finally, you can save the result. `type="probe_stack"` saves the probe of every scan position to `<fileName>_probe_stack.hdf5`, which takes much more space.

# %%
# saves the result as HDF5, plus a PNG of the final reconstruction
reconstruction.saveResults("OPR_reconstruction.hdf5", snapshot=monitor)
# reconstruction.saveResults("OPR_reconstruction", type="probe_stack")
