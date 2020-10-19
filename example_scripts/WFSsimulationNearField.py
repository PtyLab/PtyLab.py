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
simuData.Nd = 2**9/simuData.binningFactor
simuData.dxp = simuData.dxd

## define probe
probe = np.zeros((nlambda, simuData.Np, simuData.Np), dtype='np.float16')
w0 = 6e-6
wzMean = 0
for k in np.arange(nlambda):
    z0 = np.pi*w0**2/simuData.spectralDensity(k)   # Rayleigh range
    wz = w0*np.sqrt(1+(z1/simuData.zo)**2)   # beam width
    # H = circ(simuData.Xp, simuData.Yp, 2.5*wz)
    H = 1 # phase term todo: find zernike functions
    wzMean = wzMean+wz
    probe[k] = np.exp(-(simuData.Xp**2+simuData.Yp**2)/wz**2)*H
    ax1 = plt.figure(1)
    hsvplot(probe[k], ax=ax1, pixelSize=simuData.dxp, axisUnit='mm')
    plt.title('wavelength %f')



