import numpy as np
from PtyLab.utils.utils import circ, gaussian2D, cart2pol, fft2c
from PtyLab.utils.scanGrids import GenerateNonUniformFermat
from PtyLab.Operators.Operators import aspw
from PtyLab.Operators.Operators import fresnelPropagator
from PtyLab.Operators.Operators import scaledASP
from PtyLab.utils.visualisation import hsvplot, show3Dslider
import matplotlib.pylab as plt
from numpy import float128
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.misc import ascent
import os
import h5py
from PtyLab import ExperimentalData
from PtyLab import Reconstruction
from PtyLab import Monitor
from PtyLab import Params
from PtyLab import Engines
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Turn this off not to export the data at the end
export_data = False
from PtyLab.io import getExampleDataFolder

fileName = "simu"

# Set physical properties
wavelength = 632.8e-9
zo = 40e-3
nlambda = 1
npsm = 1
nosm = 1
nslice = 2  # Number of slices
dz = 9000e-06  # Slice separation distance   #kommt mir viel  #von 3500 auf 350 geändert
binningFactor = 1

# Detector coordinates
Nd = 2 ** 9
dxd = 2 ** 11 / Nd * 4.5e-6
Ld = Nd * dxd

# Probe coordinates
dxp = wavelength * zo / Ld
print(f"dxp: {wavelength * zo / Ld}")

Np = Nd
Lp = dxp * Np
xp = np.arange(-Np // 2, Np // 2) * dxp
Xp, Yp = np.meshgrid(xp, xp)
zp = 1e-2  # Pinhole-object distance

# Object coordinates
No = 2 ** 10 + 2 ** 10
dxo = dxp
Lo = dxo * No
xo = np.arange(-No // 2, No // 2) * dxo
Xo, Yo = np.meshgrid(xo, xo)

# Generate illumination
f = 20e-3  # Focal length of lens, creating a focused probe
pinhole = circ(Xp, Yp, Lp * 0.3)
pinhole = convolve2d(pinhole, gaussian2D(5, 1).astype(np.float32), mode="same")

# Propagate to lens
probe = np.copy(pinhole)
# probe = aspw(pinhole, 2 * f, wavelength, Lp)[0]

# Multiply with quadratic phase and aperture
probe = probe * np.exp(-1.0j * 2 * np.pi / wavelength * (Xp ** 2 + Yp ** 2) / (2 * f))
# probe = aspw(probe, f*1.1, wavelength, Lp)[0]

plt.figure()
plt.imshow(np.abs(probe))

# Visualize the probe
plt.figure(figsize=(5, 5))
plt.subplot(121)
plt.imshow(np.angle(probe))
plt.subplot(122)
plt.imshow(abs(probe) ** 2)
plt.title("Probe Intensity")

# Generate object
# object slice 1
d1 = 1e-3  # the smaller this parameter the larger the spatial frequencies in the simulated object
b1 = 33  # topological charge (feel free to play with this number)
theta, rho = cart2pol(Xo, Yo)
t = (1 + np.sign(np.sin(b1 * theta + 2 * np.pi * (rho / d1) ** 2))) / 2

temp = t * circ(Xo, Yo, Lo) * (1 - circ(Xo, Yo, 200 * dxo)) + circ(
    Xo, Yo, 130 * dxo
)

temp = temp.astype(complex)

obj1 = np.copy(temp)
obj1[obj1 == 0] = 0.5 + 1j
obj1 = gaussian_filter(obj1, 1)
object_slice1 = obj1

obj2 = np.copy(temp)
obj2[obj2 == 0] = 0.2 + 0.5j
obj2 = gaussian_filter(obj2, 1)

from scipy.ndimage import zoom
obj2 = zoom(ascent(), 4)
obj2 = obj2[:obj1.shape[0], :obj1.shape[1]].astype(complex)

# Shift object_slice by a certain number of pixels
shift_x = 300  # 100  # Shift in x-direction (pixels)
shift_y = 300  # 30  # Shift in y-direction (pixels)

# Create object_slice2 by shifting object_slice1
object_slice2 = np.roll(obj2, (shift_x, shift_y), axis=(0, 1))

plt.figure()
plt.subplot(121)
plt.imshow(np.abs(object_slice1))
plt.subplot(122)
plt.imshow(np.angle(object_slice1))

plt.figure()
plt.subplot(121)
plt.imshow(np.abs(object_slice2))
plt.subplot(122)
plt.imshow(np.angle(object_slice2))

print('shape object slice 1',object_slice1.shape)
print('shape object slice 2',object_slice2.shape)

# Generate positions
numPoints = 400  # Number of points
radius = 250  # Radius of final scan grid (in pixels)
p = 1  # Standard Fermat
R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)

plt.figure()
plt.plot(R, C, 'o')

# # Show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(R, C, "o")
plt.xlabel("um")
plt.title("Scan Grid")

# Optimize scan grid
encoder = np.vstack((R * dxo, C * dxo)).T
positions = np.round(encoder / dxo)
offset = np.array([50, 20])
positions = (positions + No // 2 - Np // 2 + offset).astype(int)

# Get number of positions
numFrames = len(R)
print(f"Generate positions ({numFrames})")

# Calculate estimated overlap
beamSize = (
        np.sqrt(np.sum((Xp ** 2 + Yp ** 2) * np.abs(probe) ** 2) / np.sum(abs(probe) ** 2))
        * 2.355
)

distances = np.sqrt(np.diff(R) ** 2 + np.diff(C) ** 2) * dxo
averageDistance = np.mean(distances) * 1e6
print(f"Average step size: {averageDistance:.1f} (um)")
print(f"Probe diameter: {beamSize * 1e6:.2f}")
print(f"Number of scan points: {numFrames}")

# Show scan grid on object
plt.figure(figsize=(5, 5), num=33)
ax1 = plt.axes()
hsvplot(np.squeeze(object_slice2), ax=ax1)

pos_pix = positions + Np // 2
dia_pix = beamSize / dxo
ax1.plot(
    pos_pix[:, 1],  # Corrected indexing
    pos_pix[:, 0],  # Corrected indexing
    "ro",
    alpha=0.9,
)
ax1.set_xlim(pos_pix[:, 1].min() - 100, pos_pix[:, 1].max() + 100)
ax1.set_ylim(pos_pix[:, 0].max() + 100, pos_pix[:, 0].min() - 100)

# Indicate the probe with the typical diameter
for p in pos_pix:
    c = plt.Circle((p[1], p[0]), radius=dia_pix / 2, color="black", fill=False, alpha=0.5)
    ax1.add_artist(c)
ax1.set_title("Object with Probe Positions")

## set data
entrancePupilDiameter = beamSize

# Generate ptychogram
ptychogram = np.zeros((numFrames, Nd, Nd), dtype=float128)

for loop in np.arange(numFrames):
    # Get object patch for slice 1
    # row, col = positions[loop]

    row, col = pos_pix[loop]
    sy = slice(row, row + Np)
    sx = slice(col, col + Np)
    object_patch1 = object_slice1[..., sy, sx].copy()

    # Multiply probe by the first object slice
    esw1 = object_patch1 * probe

    # Propagate the probe to the second slice
    esw1_propagated = aspw(esw1, dz, wavelength, Lp, is_FT=False)[0]

    # Get object patch for slice 2
    object_patch2 = object_slice2[..., sy, sx].copy()

    debug = False
    if debug:
        plt.figure()
        plt.imshow(np.abs(esw1_propagated))
        plt.show()

    # Multiply propagated probe by the second object slice
    esw2 = object_patch2 * esw1_propagated

    # Propagate to the detector plane
    ESW = fft2c(esw2)
    # ESW = fft2c(esw1)
    # Save data in ptychogram
    ptychogram[loop] = abs(ESW) ** 2

show3Dslider((np.log(np.abs(ptychogram)) ** 2))

exampleData = ExperimentalData(operationMode="CPM")
plt.show()

# fill up the requiredFields
exampleData.ptychogram = ptychogram
exampleData.wavelength = wavelength
exampleData.encoder = encoder
exampleData.dxd = dxd
exampleData.zo = zo
# then fill up optionalFields. If not used, set the values to None
exampleData.entrancePupilDiameter = 300e-6  # initial estimate of beam diameter
exampleData.spectralDensity = None  # used in polychromatic ptychography
exampleData.theta = None  # used in tiltPlane reflection ptychography

# call _setData to auto-calculate all the necessary variables in ExperimentalData
exampleData._setData()

## initialize the Monitor class and set values
monitor = Monitor()
monitor.figureUpdateFrequency = 5
monitor.objectPlot = "complex"  # complex abs angle
monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.5  # control object plot FoV
monitor.probeZoom = 0.5  # control probe plot FoV

## initialize the Params class and set values
params = Params()
# main parameters
params.positionOrder = "random"  # 'sequential' or 'random'
params.propagator = "Fraunhofer"  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
params.intensityConstraint = "standard"  # standard fluctuation exponential poission

# how do we want to reconstruct (all Switches are set to False by default)
params.gpuSwitch = True
params.probePowerCorrectionSwitch = False
params.modulusEnforcedProbeSwitch = False
params.comStabilizationSwitch = True
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10
params.fftshiftSwitch = False

## initialize the Reconstruction class and set values
reconstruction = Reconstruction(exampleData, params)
reconstruction.npsm = 1  # Number of probe modes to reconstruct
reconstruction.nosm = 1  # Number of object modes to reconstruct
reconstruction.nlambda = 1  # len(exampleData.spectralDensity) # Number of wavelength
reconstruction.nslice = 1  # Number of object slice
reconstruction.dz = dz
reconstruction.refrIndex = 1.0
reconstruction.initialProbe = "circ"
reconstruction.initialObject = "ones"  # upsampled, rand, circ, ones, circ_smooth, gaussian

# initialize probe and object
reconstruction.initializeObjectProbe()
# optional: customize initial probe quadratic phase

# choose the reconstruction engine and set values
engine_mPIE = Engines.mPIE(reconstruction, exampleData, params, monitor)
engine_mPIE.numIterations = 20
engine_mPIE.betaProbe = 0.25
engine_mPIE.betaObject = 0.25
# # start reconstruction
engine_mPIE.reconstruct()

probe = np.copy(reconstruction.probe[0, 0, :, 0, :, :])
obj = np.copy(reconstruction.object[0, 0, 0, 0, :, :])

reconstruction.nslice = 2  # Number of object slice
reconstruction.initializeObjectProbe()

reconstruction.probe[0, 0, :, 0, :, :] = np.copy(probe)
reconstruction.object[0, 0, 0, 0, :, :] = np.copy(obj)
params.comStabilizationSwitch = False

# # choose the reconstruction engine and set values
engine_e3PIE = Engines.e3PIE(reconstruction, exampleData, params, monitor)
engine_e3PIE.numIterations = 300
engine_e3PIE.betaProbe = 0.99
engine_e3PIE.betaObject = 0.99
# start reconstruction
engine_e3PIE.reconstruct()

reconstruction.saveResults(fileName='multi_slice_test.hdf5')
