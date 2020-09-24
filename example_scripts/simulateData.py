# This script contains a minimum working example of how to generate data
import numpy as np
from fracPy.utils.utils import circ, gaussian2D, cart2pol
from fracPy.utils.scanGrids import GenerateNonUniformFermat
from fracPy.operators.operators import aspw
from fracPy.utils.visualisation import hsvplot
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
import imageio
import tqdm
from skimage.transform import rescale
import glob
import os
import h5py
# Not implemented yet.

# create ptyLab object
simuData = ExperimentalData()

# set physical properties

simuData.wavelength = 632.8e-9
zo = 5e-2

# detector coordinates
Nd = 2**7
dxd = 2**11/Nd*4.5e-6
Ld = Nd*dxd
xd = np.arange(-Nd//2, Nd//2) * dxd
Xd, Yd = np.meshgrid(xd, xd)          # 2D coordinates in detector plane

# probe coordinates
dxp = simuData.wavelength * zo / Ld
Np = Nd
Lp = Np * dxp
xp = np.arange(-Np//2, Np//2) * dxp
Xp, Yp = np.meshgrid(xp, xp)
zp = 1e-2  # pinhole-object distance

# object coordinates
dxo = dxp
No = 2**10
Lo = No * dxo
xo = np.arange(-No//2, No//2) * dxo
Xo, Yo = np.meshgrid(xo, xo)

# generate illumination
# note: simulate focused beam
# goal: 1:1 image iris through (low-NA) lens with focal length f onto an object
f = 5e-3 # focal length of lens, creating a focused probe
pinhole = circ(Xp, Yp, Lp/2)
pinhole = convolve2d(pinhole, gaussian2D(5, 1), mode='same')

# propagate to lens
simuData.probe = aspw(pinhole, 2*f, simuData.wavelength, Lp)[0]

# multiply with quadratic phase and aperture
aperture = circ(Xp, Yp, 3*Lp/4)
aperture = convolve2d(aperture, gaussian2D(5, 3), mode='same')
simuData.probe = simuData.probe * np.exp(-1.j*2*np.pi/simuData.wavelength*(Xp**2+Yp**2)/(2*f)) * aperture
simuData.probe = aspw(simuData.probe, 2*f, simuData.wavelength, Lp)[0]

plt.figure(figsize=(5,5), num=1)
ax1 = plt.subplot(121)
hsvplot(simuData.probe, ax=ax1, pixelSize=dxp)
ax1.set_title('complex probe')
plt.subplot(122)
plt.imshow(abs(simuData.probe)**2)
plt.title('probe intensity')
plt.show(block=False)

# generate object
d = 1e-3   # the smaller this parameter the larger the spatial frequencies in the simulated object
b = 33     # topological charge (feel free to play with this number)
theta, rho = cart2pol(Xo, Yo)
t = (1 + np.sign(np.sin(b * theta + 2*np.pi * (rho/d)**2)))/2
# phaseFun = np.exp(1.j * np.atan2(Yo, Xo))
phaseFun = 1
# phaseFun = np.exp(1.j*( 1 * theta + 2*np.pi * (rho/d)**2))
t = t*circ(Xo, Yo, Lo)*(1-circ(Xo, Yo, 200*dxo))*phaseFun+circ(Xo, Yo, 130*dxo)
obj = convolve2d(t, gaussian2D(5, 3), mode='same')  # smooth edges

plt.figure(figsize=(5,5), num=2)
ax = plt.axes()
hsvplot(obj, ax=ax, pixelSize=dxo)
ax.set_title('complex probe')
plt.show(block=False)

# generate positions
# parameters
numPoints = 100   # number of points
radius = 100    # radius of final scan grid (in micrometers)
p = 1    # "clumping parameter"
# expected beam size, required to calculate overlap (expect Gaussian-like beam, derive from second moment)
beamSize = np.sqrt(np.sum((Xp**2+Yp**2)*np.abs(simuData.probe)**2)/np.sum(abs(simuData.probe)**2))*2.355
# * note:
# * p = 1 is standard Fermat
# * p > 1 yields more points towards the center of grid

# generate non-uniform Fermat grid
R, C = GenerateNonUniformFermat(numPoints, radius=radius, power=p)

# optimize scan grid
# numIterations = 5e4   # number of iterations in optimization
# print('optimize scan grid')

# prevent negative indices by centering spiral coordinates on object
R = np.round(R+No//2-Np//2+50) #todo why +50?
C = np.round(C+No//2-Np//2+50)

# get number of positions
numFrames = len(R)
print('generate positions('+str(numFrames)+')')

# show scan grid
plt.figure(figsize=(5, 5), num=99)
plt.plot(R, C, 'o')
plt.title('scan grid')

# calculate estimated overlap
distances = np.sqrt(np.diff(R)**2+np.diff(C)**2)
averageDistance = np.mean(distances)
print('average step size:%.1f (um)' % averageDistance)
print('number of scan points: %d' % numFrames)

# generate ptychogram

optimizable = Optimizable(simuData)
reconstructor = BaseReconstructor(optimizable, simuData)

ptychogram = np.zeros((Nd, Nd, numFrames))
reconstructor.propagator = 'Fresnel'
reconstructor._initializeParams()
reconstructor.object2detector()
