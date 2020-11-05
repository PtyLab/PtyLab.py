# This script contains a minimum working example of how to generate data
import numpy as np
from fracPy.utils.utils import circ, rect, posit, fft2c, ifft2c, fraccircshift
from fracPy.utils.scanGrids import GenerateConcentricGrid, GenerateRasterGrid
from fracPy.operators.operators import aspw
from fracPy.utils.visualisation import hsvplot, hsvmodeplot
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.monitors.Monitor import Monitor
from skimage.transform import rescale
import os
import h5py
from zernike import RZern

fileName = 'WFSpoly'
# create ptyLab object
simuData = ExperimentalData()

simuData.spectralDensity = 800*1e-9/np.arange(15, 31, 2)  # 9 harmonics
nlambda = len(simuData.spectralDensity)
simuData.wavelength = min(simuData.spectralDensity)
simuData.binningFactor = 1

## sample detecotr distance
simuData.zo = 198e-3
# jet-WFS distance
z1 = 80e-2
# M = 1+simuData.zo/z1

## coordinates
# detector coordinates
simuData.dxd = 13.5e-6*simuData.binningFactor
simuData.Nd = int(2**8/simuData.binningFactor)
# probe coordinates
dxp = simuData.dxd
Np = 2**11
xp = np.arange(-Np//2, Np//2)*dxp
Lp = dxp*Np
Xp, Yp = np.meshgrid(xp, xp)

## define probe
probe = np.zeros((nlambda, Np, Np), dtype=np.float32)
w0 = 8e-6
wzMean = 0
for k in np.arange(nlambda):
    z0 = np.pi*w0**2/simuData.spectralDensity[k]   # Rayleigh range
    wz = w0*np.sqrt(1+(z1/simuData.zo)**2)   # beam width
    D = 2.5*wz
    H = circ(Xp, Yp, 2.5*wz)

    cart = RZern(4)

    cart.make_cart_grid(Xp/D/2,Yp/D/2)
    c = np.zeros(cart.nk)
    c[k] = 1
    Phi = cart.eval_grid(c, matrix=True)

    plt.figure(44)
    plt.imshow(Phi)
    plt.show(block=False)


    H = 1 # phase term todo: find zernike functions
    wzMean = wzMean+wz
    probe[k] = np.exp(-(Xp**2+Yp**2)/wz**2)*H
    # plt.figure(figsize=(10,5), num=1)
    # ax1 = plt.subplot(121)
    # hsvplot(probe[k], ax=ax1, pixelSize=dxp, axisUnit='mm')
    # ax1.set_title('wavelength: %.2f nm' %(simuData.spectralDensity[k]*1e9))
    # plt.subplot(122)
    # plt.imshow(abs(probe[k]) ** 2)
    # plt.title('probe intensity')
    # plt.show(block=False)

wzMean = wzMean/nlambda
print('mean spectral probe diameter (fwhm): %.2f mm.' %(2*wzMean*1e3))
hsvmodeplot(probe)

## define WFS
pinholeDiameter = 700e-6
aperture = circ(Xp, Yp, pinholeDiameter)
f = z1 # create collimation
WFStype = 'rand'
if WFStype=='QC':
    s = 9
    numCircs = 4
    n = (pinholeDiameter/2)/dxp-2
    R,C = GenerateConcentricGrid(numCircs, s, n)
    R = R+Np//2+1
    C = C+Np//2+1
    temp = np.zeros((Np,Np), dtype=np.float32)
    for k in np.arange(len(R)):
        temp[R[k],C[k]] = 1
    WFS = temp*aperture
    WFS = convolve2d(WFS, np.ones((5, 5), dtype=int), mode='same')
elif WFStype=='rand':
    fullPeriod = 6*13.5e-6
    apertureSize = 4*13.5e-6
    WFS = 0*Xp
    n = 9
    R,C = GenerateRasterGrid(n, np.round(fullPeriod/dxp))
    print('WFS size: %d um' % (2 * max(max(np.abs(C)), max(np.abs(R))) * dxp * 1e6))
    print('WFS size: %d um' % ((max(max(R) - min(R), max(C) - min(C)) * dxp + apertureSize) * 1e6))
    R = R + Np // 2
    C = C + Np // 2

    for k in np.arange(len(R)):
        R[k] = R[k] + np.random.randint(1, 3) - 2
        C[k] = C[k] + np.random.randint(1, 3) - 2

    for k in np.arange(len(R)):
        WFS[R[k], C[k]] = 1

    subaperture = rect(Xp / apertureSize) * rect(Yp / apertureSize)
    WFS = np.abs(ifft2c(fft2c(WFS) * fft2c(subaperture)))
    WFS = WFS / np.max(WFS)

simuData.WFS = WFS[Np//2-simuData.Nd//2:Np//2+simuData.Nd//2, Np//2-simuData.Nd//2:Np//2+simuData.Nd//2]

hsvplot(simuData.WFS, pixelSize=dxp, axisUnit='mm')

## generate WFS for FIB

## generate positions
Nr = 6
s = 15
rend = 100
R, C = GenerateConcentricGrid(Nr, s, rend)
# get number of positions
numFrames = len(R)
print('generate positions (number of frames=%d)' % numFrames)
# TODO why from the second to the single to the last?
averageStep = np.mean(np.sqrt(np.diff(R[1:-1])**2+np.diff(C[1:-1])**2)) * dxp
# meanOverlap = (1-averageStep/pinholeDiameter)*fillFactor
# print('mean linear overlap: %d %%' % (meanOverlap*100))

plt.figure(figsize=(5, 5), num=3)
plt.plot(R, C, 'o-')
plt.show(block=False)

## generate ptychogram
simuData.ptychogram = np.zeros((numFrames, simuData.Nd, simuData.Nd), dtype=np.float32)
ESW = np.zeros((numFrames, Np, Np), dtype=np.complex64)

for loop in np.arange(numFrames):
    print(str(loop))

    # get object patch
    WFSshifted = np.roll(WFS, [R[loop], C[loop]], axis=(0, 1))

    # generate diffraction data (complex amplitude conversion below)
    for k in np.arange(nlambda):
        ESW[k] = aspw(probe[k]*WFSshifted, simuData.zo, simuData.spectralDensity[k], Lp)[0]

    # save data in ptychogram
    I = np.sum(abs(ESW)**2, axis=0)
    # re-shift step
    temp = posit(fraccircshift(rescale(I, 1/simuData.binningFactor, order=0), [-R[loop], -C[loop]])) # order = 0 takes the nearest-neighbor))
    simuData.ptychogram[k] =temp[Np//2-simuData.Nd//2:Np//2+simuData.Nd//2, Np//2-simuData.Nd//2:Np//2+simuData.Nd//2]
    if simuData.binningFactor>1:
        raise('binning not implemented yet')

    # inspect diffraction data

simuData.showPtychogram()

## re-define the coordinates
simuData.dxp = dxp
simuData.No = 2**10+2**9

simuData.encoder = np.vstack((R*simuData.dxo, C*simuData.dxo)).T

## simulate Poisson noise todo

## set properties
simuData.entrancePupilDiameter = pinholeDiameter
simuData.probe = []

## data inspection, check sampling requirements todo
export_data = False # exportBool in MATLAB
from fracPy.io import getExampleDataFolder
saveFilePath = getExampleDataFolder()
os.chdir(saveFilePath)
if export_data:
    hf = h5py.File(fileName+'.hdf5', 'w')
    hf.create_dataset('ptychogram', data=simuData.ptychogram, dtype='f')
    hf.create_dataset('encoder', data=simuData.encoder, dtype='f')
    hf.create_dataset('dxd', data=(simuData.dxd,), dtype='f')
    hf.create_dataset('Nd', data=(simuData.Nd,), dtype='i')
    hf.create_dataset('No', data=(simuData.No,), dtype='i')
    hf.create_dataset('zo', data=(simuData.zo,), dtype='f')
    hf.create_dataset('wavelength', data=(simuData.wavelength,), dtype='f')
    hf.create_dataset('spectralDensity', data=(simuData.spectralDensity,), dtype='f')
    hf.create_dataset('entrancePupilDiameter', data=(simuData.entrancePupilDiameter,), dtype='f')
    hf.close()
    print('An hd5f file has been saved')



