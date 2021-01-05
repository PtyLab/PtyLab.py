# This script contains a minimum working example of how to generate data
import numpy as np
from fracPy.utils.utils import circ, rect, posit, fft2c, ifft2c, fraccircshift
from fracPy.utils.scanGrids import GenerateConcentricGrid, GenerateRasterGrid
from fracPy.operators.operators import aspw
from fracPy.utils.visualisation import hsvplot, hsvmodeplot, show3Dslider
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.monitors.Monitor import Monitor
from skimage.transform import rescale
import os
import h5py

fileName = 'WFS_8_bin4'
# create ptyLab object
simuData = ExperimentalData()
harmonicNum = np.linspace(15, 29, 4)
simuData.spectralDensity = 800*1e-9/harmonicNum # 8 harmonics
nlambda = len(simuData.spectralDensity)
simuData.wavelength = min(simuData.spectralDensity)
binningFactor = 8

## sample detecotr distance
simuData.zo = 198e-3
# jet-WFS distance
z1 = 80e-2
# M = 1+simuData.zo/z1

## coordinates
# detector coordinates
dxd = 13.5e-6/binningFactor
Nd = int(2**8*binningFactor)

# probe coordinates (WFS)
dxp = dxd
Np = Nd*2
xp = np.arange(-Np//2, Np//2)*dxp
Lp = dxp*Np
Xp, Yp = np.meshgrid(xp, xp)

# object coordinates (beam)
dxo = dxp
No = Np
xo = np.arange(-No//2, No//2)*dxo
Lo = dxo*No
Xo, Yo = np.meshgrid(xo, xo)

## define probe
beam = (1 + 1j) * np.zeros((nlambda, No, No), dtype=np.float32)
w0 = 15e-6
wzMean = 0

for k in np.arange(nlambda):
    z0 = np.pi*w0**2/simuData.spectralDensity[k]   # Rayleigh range
    wz = w0*np.sqrt(1+(z1/z0)**2)   # beam width
    D = 2.5*wz

    H = 1 # phase term todo: find zernike functions
    wzMean = wzMean+wz
    # probe[k] = np.exp(-(Xo**2+Yo**2)/wz**2)*H
    beam[k] = aspw(np.exp(-(Xo ** 2 + Yo ** 2) / w0 ** 2), z1, simuData.spectralDensity[k], Lo)[0]
    # plt.figure(figsize=(10,5), num=1)
    # ax1 = plt.subplot(121)
    # hsvplot(beam[k], ax=ax1, pixelSize=dxp, axisUnit='mm')
    # ax1.set_title('wavelength: %.2f nm' %(simuData.spectralDensity[k]*1e9))
    # plt.subplot(122)
    # plt.imshow(abs(probe[k]) ** 2)
    # plt.title('probe intensity')
    # plt.show(block=False)

wzMean = wzMean/nlambda
print('mean spectral probe diameter (fwhm): %.2f mm.' %(2*wzMean*1e3))


## define WFS
pinholeDiameter = 250e-6
f = z1 # create collimation
WFStype = 'rand'
if WFStype =='QC':
    aperture = circ(Xp, Yp, pinholeDiameter)
    s = 9
    numCircs = 4
    n = (pinholeDiameter/2)/dxp-2
    R, C = GenerateConcentricGrid(numCircs, s, n)
    R = R+Np//2+1
    C = C+Np//2+1
    temp = np.zeros((Np, Np), dtype=np.float32)
    for k in np.arange(len(R)):
        temp[R[k], C[k]] = 1
    WFS = temp*aperture
    WFS = convolve2d(WFS, np.ones((5, 5), dtype=int), mode='same')
elif WFStype == 'rand':
    aperture = rect(Xp, pinholeDiameter/2)*rect(Yp, pinholeDiameter/2)
    # fullPeriod = 6*13.5e-6
    # apertureSize = 3*13.5e-6
    fullPeriod = 12e-6
    apertureSize = 6e-6
    WFS = 0*Xp
    n = int(pinholeDiameter//fullPeriod)
    R, C = GenerateRasterGrid(n, np.round(fullPeriod/dxp))
    print('WFS size: %d um' % (2 * max(max(np.abs(C)), max(np.abs(R))) * dxp * 1e6))
    print('WFS size: %d um' % ((max(max(R) - min(R), max(C) - min(C)) * dxp + apertureSize) * 1e6))
    R = R + Np // 2
    C = C + Np // 2

    np.random.seed(1)
    R_offset = np.random.randint(1, 3, len(R))
    np.random.seed(2)
    C_offset = np.random.randint(1, 3, len(C))
    R = R+R_offset-2
    C = C+C_offset-2

    for k in np.arange(len(R)):
        WFS[R[k], C[k]] = 1

    subaperture = rect(Xp / apertureSize) * rect(Yp / apertureSize)
    WFS = np.abs(ifft2c(fft2c(WFS) * fft2c(subaperture)))  # convolution of the subaperture with the scan grid
    WFS = WFS / np.max(WFS)*aperture

# simuData.WFS = WFS[Np//2-Nd//2:Np//2+Nd//2, Np//2-Nd//2:Np//2+Nd//2]
simuData.WFS = WFS
# number of non-zero pixels
nnzPixels = np.sum(WFS)
fillFactor = nnzPixels / np.sum(aperture)
hsvplot(WFS, pixelSize=dxp, axisUnit='mm')
plt.show(block=False)


## generate WFS for FIB
N = int(pinholeDiameter/dxp)+20
WFScrop = WFS[Np//2-N//2:Np//2+N//2,Np//2-N//2:Np//2+N//2]
FIBpixelSize = 135e-9
upsamplingFactor = dxp / FIBpixelSize
WFScrop = rescale(WFScrop, upsamplingFactor, order=0)
WFScrop[WFScrop> 0.1] = 1
hsvplot(WFScrop, pixelSize=dxp, axisUnit='mm')
plt.show(block=False)

## generate scan positions
Nr = 6
s = 4
rend = 20
R, C = GenerateConcentricGrid(Nr, s, rend)
# get number of positions
numFrames = len(R)
print('generate positions (number of frames=%d)' % numFrames)
# TODO why from the second to the single to the last?
averageStep = np.mean(np.sqrt(np.diff(R[1:-1])**2+np.diff(C[1:-1])**2)) * dxp
meanOverlap = (1-averageStep/pinholeDiameter)*fillFactor
print('mean linear overlap: %d %%' % (meanOverlap*100))

plt.figure(figsize=(5, 5), num=3)
plt.plot(R*dxp*1e3, C*dxp*1e3, 'o-')
plt.xlabel('mm')
plt.show()

## generate ptychogram
ptychogram = np.zeros((numFrames, Nd, Nd), dtype=np.float32)
ESW = np.zeros((nlambda, No, No), dtype=np.complex64)

for loop in np.arange(numFrames):
    print(str(loop))

    # get object patch
    WFSshifted = np.roll(WFS, [R[loop], C[loop]], axis=(0, 1))

    # generate diffraction data (complex amplitude conversion below)
    for k in np.arange(nlambda):
        ESW[k] = aspw(beam[k] * WFSshifted, simuData.zo, simuData.spectralDensity[k], Lp)[0]

    # save data in ptychogram
    I = np.sum(abs(ESW)**2, axis=0)
    # re-shift step
    temp = posit(fraccircshift(I, [-R[loop], -C[loop]]))
    ptychogram[loop] =temp[Np//2-Nd//2:Np//2+Nd//2, Np//2-Nd//2:Np//2+Nd//2]

if binningFactor > 1:
    simuData.ptychogram = np.sum(ptychogram.reshape(numFrames, Nd//binningFactor, binningFactor, Nd//binningFactor, binningFactor), axis=(-1, -3))
else:
    simuData.ptychogram = ptychogram

# inspect diffraction data
simuData.showPtychogram()

## re-define the coordinates
simuData.dxd = dxd*binningFactor
simuData.Nd = simuData.ptychogram.shape[-1]
simuData.encoder = np.vstack((R*dxp, C*dxp)).T
simuData.No = 2**10+2**9

## simulate Poisson noise
bitDepth = 14
maxNumCountsPerDiff = 2**bitDepth

# normalize data (ptychogram)
simuData.ptychogram = simuData.ptychogram/np.max(simuData.ptychogram) * maxNumCountsPerDiff
ptychogram_noNoise = simuData.ptychogram.copy()

# simulate Poisson noise
noise = np.random.poisson(simuData.ptychogram)
simuData.ptychogram += noise
simuData.ptychogram[simuData.ptychogram < 0] = 0

# compare noiseless data noisy
ptychogram_comparison = np.concatenate((ptychogram_noNoise,simuData.ptychogram), axis=1)
show3Dslider(np.log(ptychogram_comparison+1))

## set properties
simuData.entrancePupilDiameter = pinholeDiameter

## data inspection, check sampling requirements todo
export_data = False #
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