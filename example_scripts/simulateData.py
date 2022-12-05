# This script contains a minimum working example of how to generate data
# todo: mention that it generates simu.hdf5
import numpy as np
from PtyLab.utils.utils import circ, gaussian2D, cart2pol
from PtyLab.utils.scanGrids import GenerateNonUniformFermat
from PtyLab.Operators.Operators import aspw
from PtyLab.utils.visualisation import hsvplot, show3Dslider
import matplotlib.pylab as plt
from scipy.signal import convolve2d
import os
import h5py

fileName = "simuRecent_copy"

# set physical properties

wavelength = 632.8e-9
zo = 5e-2
nlambda = 1
npsm = 1
nosm = 1
nslice = 1
binningFactor = 1

# detector coordinates
Nd = 2**7
dxd = 2**11 / Nd * 4.5e-6
Ld = Nd * dxd

# probe coordinates
dxp = wavelength * zo / Ld
Np = Nd
Lp = dxp * Np
xp = np.arange(-Np // 2, Np // 2) * dxp
Xp, Yp = np.meshgrid(xp, xp)
zp = 1e-2  # pinhole-object distance

# object coordinates
No = 2**10 + 2**9
dxo = dxp
Lo = dxo * No
xo = np.arange(-No // 2, No // 2) * dxo
Xo, Yo = np.meshgrid(xo, xo)

# generate illumination
# note: simulate focused beam
# goal: 1:1 image iris through (low-NA) lens with focal length f onto an object
f = 5e-3  # focal length of lens, creating a focused probe
pinhole = circ(Xp, Yp, Lp / 2)
pinhole = convolve2d(pinhole, gaussian2D(5, 1).astype(np.float32), mode="same")

# propagate to lens
probe = aspw(pinhole, 2 * f, wavelength, Lp)[0]

# multiply with quadratic phase and aperture
aperture = circ(Xp, Yp, 3 * Lp / 4)
aperture = convolve2d(aperture, gaussian2D(5, 3).astype(np.float32), mode="same")
probe = (
    probe
    * np.exp(-1.0j * 2 * np.pi / wavelength * (Xp**2 + Yp**2) / (2 * f))
    * aperture
)
probe = aspw(probe, 2 * f, wavelength, Lp)[0]


plt.figure(figsize=(5, 5), num=1)
ax1 = plt.subplot(121)
hsvplot(probe, ax=ax1, pixelSize=dxp)
ax1.set_title("complex probe")
plt.subplot(122)
plt.imshow(abs(probe) ** 2)
plt.title("probe intensity")
plt.show(block=False)

# generate object
d = 1e-3  # the smaller this parameter the larger the spatial frequencies in the simulated object
b = 33  # topological charge (feel free to play with this number)
theta, rho = cart2pol(Xo, Yo)
t = (1 + np.sign(np.sin(b * theta + 2 * np.pi * (rho / d) ** 2))) / 2
# phaseFun = np.exp(1.j * np.arctan2(Yo, Xo))
phaseFun = 1
# phaseFun = np.exp(1.j*( 2* theta + 2*np.pi * (rho/d)**2))
t = t * circ(Xo, Yo, Lo) * (1 - circ(Xo, Yo, 200 * dxo)) * phaseFun + circ(
    Xo, Yo, 130 * dxo
)
obj = convolve2d(t, gaussian2D(5, 3), mode="same")  # smooth edges
# obj_phase = np.exp(1.j*2*np.pi/wavelength*(Xo**2+Yo**2)*20)
object = obj * phaseFun

plt.figure(figsize=(5, 5), num=2)
ax = plt.axes()
hsvplot(np.squeeze(object), ax=ax)
ax.set_title("complex probe")
plt.show(block=False)

# generate positions
# generate non-uniform Fermat grid
# parameters
numPoints = 100  # number of points
radius = 100  # radius of final scan grid (in pixels)
p = 1  # p = 1 is standard Fermat;  p > 1 yields more points towards the center of grid
R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)
# show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(R, C, "o")
plt.xlabel("um")
plt.title("scan grid")
plt.show(block=False)

# Todo: Add comments here or in a help file

# optimize scan grid
# numIterations = 5e4   # number of iterations in optimization
# print('optimize scan grid')
encoder = np.vstack((R * dxo, C * dxo)).T

# prevent negative indices by centering spiral coordinates on object
positions = np.round(encoder / dxo)
offset = 50
positions = (positions + No // 2 - Np // 2 + offset).astype(int)

# get number of positions
numFrames = len(R)
print("generate positions(" + str(numFrames) + ")")

# calculate estimated overlap
# expected beam size, required to calculate overlap (expect Gaussian-like beam, derive from second moment)
beamSize = (
    np.sqrt(np.sum((Xp**2 + Yp**2) * np.abs(probe) ** 2) / np.sum(abs(probe) ** 2))
    * 2.355
)
distances = np.sqrt(np.diff(R) ** 2 + np.diff(C) ** 2) * dxo
averageDistance = np.mean(distances)
print("average step size:%.1f (um)" % averageDistance)
print("number of scan points: %d" % numFrames)

# show scan grid on object
plt.figure(figsize=(5, 5), num=33)
ax1 = plt.axes()
hsvplot(np.squeeze(object), ax=ax1)
ax1.plot(positions[0, 1] + Np // 2, positions[0, 0] + Np // 2, ".")
ax1.set_title("complex probe")
plt.show(block=False)

# set data
entrancePupilDiameter = beamSize

# generate ptychogram
ptychogram = np.zeros((numFrames, Nd, Nd))

propagator = "Fraunhofer"
for loop in np.arange(numFrames):
    # get object patch
    row, col = positions[loop]
    sy = slice(row, row + Np)
    sx = slice(col, col + Np)
    # note that object patch has size of probe array
    objectPatch = object[..., sy, sx].copy()
    # multiply each probe mode with object patch
    esw = objectPatch * probe
    # generate diffraction data, propagate the esw to the detector plane
    ESW = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(esw), norm="ortho"))
    # save data in ptychogram
    ptychogram[loop] = abs(ESW) ** 2
    # inspect diffraction data

# calcuate noise
## simulate Poisson noise
bitDepth = 14
maxNumCountsPerDiff = 2**bitDepth

# normalize data (ptychogram)
ptychogram = ptychogram / np.max(ptychogram) * maxNumCountsPerDiff
ptychogram_noNoise = ptychogram.copy()

# simulate Poisson noise
noise = np.random.poisson(ptychogram)
ptychogram += noise
ptychogram[ptychogram < 0] = 0

# compare noiseless data noisy
ptychogram_comparison = np.concatenate((ptychogram_noNoise, ptychogram), axis=1)
show3Dslider(np.log(ptychogram_comparison + 1))


# export data
export_data = False  # exportBool in MATLAB
from PtyLab.io import getExampleDataFolder

saveFilePath = getExampleDataFolder()
os.chdir(saveFilePath)
if export_data:
    hf = h5py.File(fileName + ".hdf5", "w")
    hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
    hf.create_dataset("encoder", data=encoder, dtype="f")
    hf.create_dataset("binningFactor", data=binningFactor, dtype="i")
    hf.create_dataset("dxd", data=(dxd,), dtype="f")
    hf.create_dataset("Nd", data=(Nd,), dtype="i")
    hf.create_dataset("No", data=(No,), dtype="i")
    hf.create_dataset("zo", data=(zo,), dtype="f")
    hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
    hf.create_dataset("entrancePupilDiameter", data=(entrancePupilDiameter,), dtype="f")
    hf.close()
    print("An hd5f file has been saved")
