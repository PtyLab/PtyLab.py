# %% [markdown]
# # CPM Differentiable Ptychography
#
# This example reconstructs measured soft X-ray data with `GradientEngine`, the optional PyTorch engine. Instead of hand-derived update rules, it writes down a forward model and a loss, lets PyTorch compute the gradients, and updates the object and probe with an optimizer. It covers only the basic workflow; the [Differentiable Ptychography](../differentiable/overview.md) guide describes every option.
#
# **Installation:** `GradientEngine` runs on PyTorch. Install PtyLab with the `torch` extra. On Linux and Windows, `--torch-backend=cu126` picks the CUDA 12 build that matches CuPy; drop it on macOS (see [Installation](../getting-started/installation.md#differentiable-ptychography)):
#
# ```bash
# uv pip install -U "ptylab[gpu,torch]" --torch-backend=cu126
# ```
#
# **What you'll learn:**
#
# - How to create a `GradientEngine` from the usual PtyLab objects
# - How to recover object and probe from a flat start with `blind_recipe`
# - How to reconstruct several probe modes (mixed states), and when object states help
# - How to choose the estimated parameters and switch the loss
# - How to let the probe vary across the scan with orthogonal probe relaxation (OPR)
#
# !!! note "Dataset"
#     The datasets are downloaded into `example_data/` at the project root on the first run. Most sections use `helicalbeam.hdf5` (about 350 MB), the dataset of the [CPM Mixed States](tutorial_mixedStates.md) example, from L. Loetgering et al., "Generation and characterization of focused helical x-ray beams," Sci. Adv. 6, eaax8836 (2020). The last section uses `OPR_dp_processed.hdf5` (about 180 MB), the dataset of the [CPM Orthogonal Probe Relaxation](tutorial_OPR.md) example.
#
# !!! tip "Prerequisites"
#     Complete [CPM Example Data](tutorial_CPM.md) before this example. [CPM Mixed States](tutorial_mixedStates.md) explains the mixed-state model in more depth.

# %%
import urllib.request
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import PtyLab
from PtyLab.Engines.GradientEngine import (
    AbsorbingObject,
    AbsorbingProbeBoundary,
    GradientEngine,
    OPRProjection,
    ProbeCentering,
    ProbeModeOrthogonalization,
    amplitude_loss,
    blind_recipe,
)
from PtyLab.io import getExampleDataFolder
from PtyLab.utils.visualisation import hsvplot

fileName = "helicalbeam.hdf5"
filePath = getExampleDataFolder() / fileName

if not filePath.exists():
    print(f"Downloading {fileName}..")
    urllib.request.urlretrieve(
        "https://ndownloader.figshare.com/files/38419391", filePath
    )
    print("Download complete.")
else:
    print(f"{fileName} found, skipping download.")

# %% [markdown]
# ## Set up the data and the initial guesses
#
# `GradientEngine` uses the same `ExperimentalData`, `Reconstruction`, `Params` and `Monitor` as the classic engines. The data are loaded as in the [CPM Mixed States](tutorial_mixedStates.md) example: orientation `4` (a transpose), and every second scan position, which halves the run time and leaves enough overlap.

# %%
experimentalData = PtyLab.ExperimentalData(filePath, operationMode="CPM")
experimentalData.setOrientation(4)
# keep every second scan position; _setData() updates the number of frames
experimentalData.ptychogram = experimentalData.ptychogram[::2]
experimentalData.encoder = experimentalData.encoder[::2]
experimentalData._setData()

params = PtyLab.Params()
params.propagatorType = "Fresnel"  # the detector is in the Fresnel regime

# %% [markdown]
# The function below creates a fresh reconstruction with a flat object (`ones`) and `npsm` copies of a `circ` probe, each with a small random perturbation of its own. The single-mode and the mixed-state runs below then start from the same guess.


# %%
def make_reconstruction(npsm):
    reconstruction = PtyLab.Reconstruction(experimentalData, params)
    reconstruction.npsm = npsm  # number of probe modes
    reconstruction.initialProbe = "circ"
    experimentalData.entrancePupilDiameter = reconstruction.Np / 3 * reconstruction.dxp
    reconstruction.initialObject = "ones"
    np.random.seed(0)  # same perturbations of the initial guess in every run
    reconstruction.initializeObjectProbe()
    return reconstruction


# %% [markdown]
# ## Forward model and loss
#
# For scan position $j$, the model predicts the detector intensity
#
# $$
# I_j(\mathbf q) = \left|\mathcal{D}\{O(\mathbf r + \mathbf x_j)\,P(\mathbf r)\}\right|^2 ,
# $$
#
# where $O$ is the object, $P$ the probe, $\mathbf x_j$ the scan position, $\mathcal D$ the propagation to the detector (here Fresnel) and $\mathbf q$ the detector pixel. The default loss compares predicted and measured amplitudes over all pixels $i$ of all frames:
#
# $$
# \mathcal{L} = \frac{1}{P_\text{tot}}\sum_i \left(\sqrt{I_i} - \sqrt{y_i}\right)^2 ,
# $$
#
# where $y_i$ is the measured intensity and $P_\text{tot} = \sum_i y_i$ the total measured power. PyTorch computes $\partial\mathcal L/\partial O$ and $\partial\mathcal L/\partial P$, and the optimizer takes one step per iteration over the full scan. The error plotted below is $\mathcal L$.
#
# The engine is created like the classic engines. It runs on the GPU through PyTorch when CUDA is available, and on the CPU otherwise.

# %%
single = make_reconstruction(npsm=1)

monitor = PtyLab.Monitor()
monitor.figureUpdateFrequency = 100  # redraw the live figure every 100 iterations
monitor.objectPlot = "abs"

engine = GradientEngine(single, experimentalData, params, monitor)
print("device:", engine.device)

# %% [markdown]
# ## Blind reconstruction with `blind_recipe`
#
# From a flat start, plain Adam can settle in a poor solution. `blind_recipe` builds two phases, run with `engine.reconstruct(phases=...)`:
#
# 1. The first 20 % of the iterations use a PIE-style preconditioned gradient descent, which finds the right basin.
# 2. The rest use Adam on the engine's loss.
#
# `ProbeCentering` keeps the probe's centre of mass in the middle of its window after each step, like `params.comStabilizationSwitch` in the classic engines.

# %%
history = engine.reconstruct(
    phases=blind_recipe(iterations=300, constraints=[ProbeCentering()])
)

# %%
# reconstruction.error holds the loss of every iteration of both phases
boundary = len(history[0]["error"])
plt.semilogy(single.error)
plt.axvline(boundary, color="gray", linestyle="--", label="preconditioned → Adam")
plt.xlabel("iteration")
plt.ylabel("loss $\\mathcal{L}$")
plt.legend()
plt.show()

# %% [markdown]
# The estimates are written back to `reconstruction.object` and `reconstruction.probe` as NumPy arrays, in the 6-D layout of `Reconstruction` `(nlambda, nosm, npsm, nslice, N, N)`.


# %%
def show_estimates(reconstruction, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].imshow(np.abs(reconstruction.object[0, 0, 0, 0]), cmap="gray")
    axes[0].set_title("object amplitude")
    axes[0].axis("off")
    # hue = phase, brightness = amplitude, as in the monitor
    hsvplot(reconstruction.probe[0, 0, 0, 0], ax=axes[1])
    axes[1].set_title("probe (first mode)")
    fig.suptitle(title)
    plt.show()


show_estimates(single, "1 probe mode")

# %% [markdown]
# The object is blurred: the beam was partially coherent, and one coherent probe cannot fit these data.
#
# ## Mixed states
#
# A partially coherent beam, or a sample that fluctuates during one exposure (for example by vibrating), records an incoherent sum of diffraction patterns. The mixed-state model describes it with probe modes $P_m$ and object states $O_a$, whose intensities add:
#
# $$
# I_j(\mathbf q) = \sum_{a}\sum_{m} \left|\mathcal{D}\{O_a(\mathbf r + \mathbf x_j)\,P_m(\mathbf r)\}\right|^2 .
# $$
#
# Set `reconstruction.npsm` (probe modes) and `reconstruction.nosm` (object states) before `initializeObjectProbe()`; PyTorch differentiates through the sum. The states must start different from each other, since identical states get identical gradients and never separate; the random perturbation in `make_reconstruction` takes care of that.
#
# Only the summed intensity is fixed by the data, so mixtures of the states fit equally well. `ProbeModeOrthogonalization(step=10)` and `ObjectStateOrthogonalization(step=10)` rotate them into orthogonal states sorted by power, like `params.orthogonalizationSwitch` in the classic engines, without changing the fit. With both probe and object states, such constraints are needed for a unique result ([Li et al., 2016](https://doi.org/10.1364/OE.24.009038)). The sample here is static, so this example uses four probe modes and `nosm = 1`.

# %%
mixed = make_reconstruction(npsm=4)

# a new monitor, so its figure shows this run
monitor = PtyLab.Monitor()
monitor.figureUpdateFrequency = 100
monitor.objectPlot = "abs"

engine = GradientEngine(mixed, experimentalData, params, monitor)
constraints = [ProbeCentering(), ProbeModeOrthogonalization(step=10)]
history = engine.reconstruct(
    phases=blind_recipe(iterations=300, constraints=constraints)
)

# %%
plt.semilogy(single.error, label="1 mode")
plt.semilogy(mixed.error, label="4 modes")
plt.xlabel("iteration")
plt.ylabel("loss $\\mathcal{L}$")
plt.legend()
plt.show()

# %% [markdown]
# With four modes the model also reproduces the incoherent part of the patterns, and the loss ends lower. `engine.model.probe.mode_powers()` returns the power $s_m = \sum_{\mathbf r} |P_m(\mathbf r)|^2$ of each mode, shape `(nlambda, npsm)`. Its fractions $w_m = s_m / \sum_k s_k$ show how much of the beam each mode carries.

# %%
power = engine.model.probe.mode_powers()[0].cpu().numpy()  # (npsm,)
weights = power / power.sum()
print("mode weights:", np.round(weights, 3))

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout="constrained")
for k, ax in enumerate(axes):
    hsvplot(mixed.probe[0, 0, k, 0], ax=ax, pixelSize=mixed.dxp, axisUnit="um")
    ax.set_title(f"mode {k + 1}: {100 * weights[k]:.1f} %")
    if k > 0:
        ax.set_ylabel("")  # keeps the label clear of the previous colorbar
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
for ax, rec, title in zip(axes, [single, mixed], ["1 mode", "4 modes"]):
    ax.imshow(np.abs(rec.object[0, 0, 0, 0]), cmap="gray")
    ax.set_title(f"object amplitude, {title}")
    ax.axis("off")
plt.show()

# %% [markdown]
# With four modes the edges and the grating are sharp, and the dominant mode carries about 70 % of the power, as with mPIE in the [CPM Mixed States](tutorial_mixedStates.md) example.
#
# ## Choose what to estimate and which loss
#
# `engine.parameters` selects the estimated tensors and their learning rates; tensors that are not listed stay fixed. The probe rate is in scale-free units (the probe is stored divided by its typical amplitude), so the same value works across photon counts.
#
# `engine.lossFunction` sets the loss, with its options passed through `functools.partial`. Below, the 4-mode reconstruction continues with the Anscombe version of the amplitude loss, which suits photon-counting (Poisson) data better:
#
# $$
# \mathcal{L}_\text{A} = \frac{1}{P_\text{tot}}\sum_i \left(\sqrt{I_i + c} - \sqrt{y_i + c}\right)^2 ,\qquad c = \tfrac{3}{8} ,
# $$
#
# where the shift $c$ makes the square root of Poisson counts have nearly constant variance. The Poisson and mixed Poisson-Gaussian losses are set the same way; see the [guide](../differentiable/overview.md#losses). Each `reconstruct()` starts a fresh optimizer from the current estimates. `reconstruct(phases=...)` restores the engine's own settings when it ends, so the constraints of the phases are set on the engine here.

# %%
engine.parameters = {"object": {"lr": 0.03}, "probe": {"lr": 0.1}}
engine.lossFunction = partial(amplitude_loss, anscombe_offset_switch=True)
engine.constraints = constraints
engine.numIterations = 200
engine.reconstruct()

show_estimates(mixed, "4 probe modes, after 200 more iterations with the Anscombe loss")

# %%
## save the result as HDF5, plus a PNG of the monitor figure, like with the classic engines
mixed.saveResults("helicalbeam_gradient.hdf5", snapshot=monitor)


# %% [markdown]
# ## Orthogonal probe relaxation
#
# The last section reconstructs the drifting EUV beam of the [CPM Orthogonal Probe Relaxation](tutorial_OPR.md) example, which explains the model. The data and the geometry are loaded as there. Every scan position gets its own probe, so this run needs a GPU with about 16 GB; a smaller `engine.batchSize`, such as 64, lowers the peak.

# %%
fileName = "OPR_dp_processed.hdf5"
filePath = getExampleDataFolder() / fileName

if not filePath.exists():
    print(f"Downloading {fileName}..")
    urllib.request.urlretrieve(
        "https://ndownloader.figshare.com/files/38461082", filePath
    )
    print("Download complete.")
else:
    print(f"{fileName} found, skipping download.")

# %%
experimentalData = PtyLab.ExperimentalData(filePath, operationMode="CPM")
experimentalData.setOrientation(1)
experimentalData.zo = 31.0e-3  # sample-detector distance
experimentalData.dxd = 2 * 13.5e-6  # detector pixel size, 13.5 um pixels binned 2 x 2
experimentalData.wavelength = 13.5e-9
experimentalData._setData()

params = PtyLab.Params()
params.propagatorType = "Fraunhofer"

reconstruction = PtyLab.Reconstruction(experimentalData, params)
reconstruction.No = 2000  # object array size, in pixels
reconstruction.npsm = 6  # number of probe modes
reconstruction.initialProbe = "circ"
experimentalData.entrancePupilDiameter = 10e-6
reconstruction.initialObject = "ones"
np.random.seed(0)
reconstruction.initializeObjectProbe()

monitor = PtyLab.Monitor()
monitor.figureUpdateFrequency = 100
monitor.objectPlot = "complex"
monitor.objectZoom = 2
monitor.probeZoom = 0.1

# %% [markdown]
# First a shared probe from the flat start, with `blind_recipe`. The constraints replace the switches of the OPR example: `ProbeModeOrthogonalization` for `orthogonalizationSwitch`, `AbsorbingProbeBoundary` for `absorbingProbeBoundary` and `AbsorbingObject` for `absObjectSwitch`.

# %%
constraints = [
    ProbeModeOrthogonalization(step=10),
    AbsorbingProbeBoundary(),
    AbsorbingObject(beta=0.01),
]
engine = GradientEngine(reconstruction, experimentalData, params, monitor)
history = engine.reconstruct(
    phases=blind_recipe(iterations=200, constraints=constraints)
)
object_mixed = reconstruction.object.copy()  # mixed states, one probe for all positions

# %% [markdown]
# Then OPR. `relaxProbe()` copies the probe to every scan position, and `OPRProjection` keeps the stack in a subspace of `rank` basis probes after every step, blended with `relax` (`params.OPR_subspace` and `params.OPR_alpha` of the `OPR` engine). All six modes vary. The PIE-like preconditioned optimizer takes small steps where the object is dimly lit, which keeps the edge of the scan cleaner than Adam does.

# %%
engine_OPR = engine.relaxProbe()  # the same engine, now with one probe per position
engine_OPR.optimizerType = "preconditioned"
engine_OPR.parameters = {"object": {"lr": 0.3}, "probe": {"lr": 0.3}}
engine_OPR.constraints = [OPRProjection(rank=5, relax=0.05), *constraints]
engine_OPR.numIterations = 200
engine_OPR.reconstruct()

# %% [markdown]
# The probes are written to `reconstruction.probe_stack`, in the layout of the `OPR` engine `(nlambda, nosm, npsm, nslice, Np, Np, numFrames)`, and `reconstruction.probe` holds their dominant component. The power of the first mode at each position shows the drift during the scan.

# %%
stack = reconstruction.probe_stack[0, 0, 0, 0]  # mode 1: (Np, Np, numFrames)
power = np.sum(np.abs(stack) ** 2, axis=(0, 1))
plt.plot(power / power.mean())
plt.xlabel("frame index")
plt.ylabel("relative power of mode 1")
plt.show()

# %%
crop = (slice(400, 1600), slice(400, 1600))  # the scanned area
object_OPR = reconstruction.object[0, 0, 0, 0][crop]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
for ax, obj, title in zip(
    axes,
    [object_mixed[0, 0, 0, 0][crop], object_OPR],
    ["mixed states", "mixed states + OPR"],
):
    obj = obj * np.exp(1j * np.angle(np.vdot(obj, object_OPR)))  # same global phase
    # brightness saturates at twice the median amplitude of the centre
    scale = 2 * np.median(np.abs(obj[300:900, 300:900])) / np.abs(obj).max()
    hsvplot(obj, ax=ax, amplitudeScalingFactor=scale)
    ax.set_title(f"object, {title} (6 modes)")
    ax.axis("off")
plt.show()
