# This script contains a minimum working example of how to generate data
import numpy as np
from fracPy.utils.utils import circ, gaussian2D, cart2pol
from fracPy.utils.scanGrids import GenerateConcentricGrid
from fracPy.operators.operators import aspw
from fracPy.utils.visualisation import hsvplot
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.monitors.Monitor import Monitor
import os
import h5py

fileName = 'WFSpoly'
# create ptyLab object
simuData = ExperimentalData()

simuData.spectralDensity = 1030*1e-9/np.arange(51, 84, 4)  # 9 harmonics
nlambda = len(simuData.spectralDensity)
simuData.wavelength = min(simuData.spectralDensity)
simuData.binningFactor = 1

## sample detecotr distance
simuData.zo = 515e-3
z1 = 1097.3e-3-simuData.zo
M = 1+simuData.zo/z1

## coordinates
simuData.dxd = 10e-6*simuData.binningFactor/M
simuData.Nd = int(2**9/simuData.binningFactor)
simuData.dxp = simuData.dxd

## define probe
probe = np.zeros((nlambda, simuData.Np, simuData.Np), dtype=np.float32)
w0 = 1e-4
wzMean = 0
for k in np.arange(nlambda):
    z0 = np.pi*w0**2/simuData.spectralDensity[k]   # Rayleigh range
    wz = w0*np.sqrt(1+(z1/simuData.zo)**2)   # beam width
    # H = circ(simuData.Xp, simuData.Yp, 2.5*wz)
    H = 1 # phase term todo: find zernike functions
    wzMean = wzMean+wz
    probe[k] = np.exp(-(simuData.Xp**2+simuData.Yp**2)/wz**2)*H
    plt.figure(figsize=(10,5), num=1)
    # ax1 = plt.subplot(121)
    # hsvplot(probe[k], ax=ax1, pixelSize=simuData.dxp, axisUnit='mm')
    # ax1.set_title('wavelength: %.2f nm' %(simuData.spectralDensity[k]*1e9))
    # plt.subplot(122)
    # plt.imshow(abs(probe[k]) ** 2)
    # plt.title('probe intensity')
    # plt.show(block=False)

wzMean = wzMean/nlambda
print('mean spectral probe diameter (fwhm): %.2f mm.' %(2*wzMean*1e3))

## define WFS
pinholeDiameter = 600e-6
obj = circ(simuData.Xp, simuData.Yp, pinholeDiameter)
f = z1
# concentric grid
s = 9
n = (pinholeDiameter/2)/simuData.dxp-2
R,C = GenerateConcentricGrid(8, s, n)

