# This script contains some example of preprocessing an dataset into a hdf5 file that can be read into the
# fracPy dataset.
import numpy as np
import matplotlib.pylab as plt
import imageio
import tqdm
from skimage.transform import rescale
import glob
import os
import h5py


filePath = r"D:/fracmat/ptyLab/lenspaper4/AVT camera (GX1920)"
os.chdir(filePath)

fileName = 'recent'
# wavelength
wavelength = 450e-9
# binning
binningFactor = 4
# padding for superresolution
padFactor = 1
# set magnification if any objective lens is used
magfinication = 1
# object detector distance  (initial guess)
zo = 19.23e-3
# set detection geometry
# A: camera to closer side of stage (allows to bring camera close in transmission)
# B: camera to further side of stage (doesn't allow to bring close in transmission),
# or other way around in reflection
# C: objective + tube lens in transmission
measurementMode = 'A'
# camera
camera = 'GX'
if camera == 'GX':
    N = 1456
    M = 1456
    dxd = 4.54e-6 * binningFactor / magfinication # effective detector pixel size is magnified by binning
    backgroundOffset = 30 # globally subtracted from raw data (diffraction intensities), play with this value
elif camera == 'Hamamatsu':
    N = 2**11
    M = 2**11
    dxd = 6.5e-6 * binningFactor / magfinication
    backgroundOffset = 30

# number of frames is calculated automatically
framesList = glob.glob('*'+'.tif')
framesList.sort()
numFrames = len(framesList)-1

# read background
dark = imageio.imread('background.tif')

# read empty beam (if available)

# binning
ptychogram = np.zeros((numFrames, N//binningFactor*padFactor, N//binningFactor*padFactor),
                      dtype='float32')

# read frames
pbar = tqdm.trange(numFrames, leave=True)
for k in pbar:
    # get file name
    pbar.set_description('reading frame' + framesList[k])
    temp = imageio.imread(framesList[k])-dark-backgroundOffset
    temp[temp < 0] = 0  #todo check if data type is single
    # crop
    temp = temp[M//2-N//2:M//2+N//2-1, M//2-N//2:M//2+N//2-1]
    # binning
    temp = rescale(temp, 1/binningFactor, order=0) # order = 0 takes the nearest-neighbor
    # flipping
    if measurementMode == 'A':
        temp = np.flipud(temp)
    elif measurementMode == 'B':
        temp = np.rot90(temp, axes=(0, 1))
    elif measurementMode == 'C':
        temp = np.rot90(np.flipud(temp), axes=(0, 1))

    # zero padding
    ptychogram[k] = np.pad(temp, (padFactor-1)*N//binningFactor//2)

# set experimental specifications:
entrancePupilDiameter = 1000e-6

# detector coordinates
Nd = ptychogram.shape[-1]              # number of detector pixels
Ld = Nd * dxd                         # effective size of detector
xd = np.arange(-Nd//2, Nd//2) * dxd   # 1D coordinates in detector plane
Xd, Yd = np.meshgrid(xd, xd)          # 2D coordinates in detector plane

# object coordinates
dxo = wavelength * zo / Ld          # Fraunhofer/ Fresnel
# dxo = dxd                         # asp
# dxo = 400e-9 * zo /Ld             # scaled asp, choose freely (be careful not to depart too much from Fraunhofer condition)
No = 2**12
Lo = No * dxo
xo = np.arange(-No//2, No//2) * dxo
Xo, Yo = np.meshgrid(xo, xo)

# probe coordinates
dxp = dxo
Np = Nd
Lp = Np * dxp
xp = np.arange(-Np//2, Np//2) * dxp
Xp, YP = np.meshgrid(xp, xp)

# get positions
# get file name (this assumes there is only one text file in the raw data folder)
positionFileName = glob.glob('*'+'.txt')[0]

# take raw data positions
T = np.genfromtxt(positionFileName, delimiter=' ', skip_header=2)
# convert to micrometer
encoder = (T-T[0]) * 1e-6
# encoder = (T-T[0]) * 1e-6 / magfinication
# convert into pixels
position0 = np.round(encoder / dxo)
# center within object grid
position0 = position0 + No // 2 - Np // 2
# take only the frames needed (if numFrames smaller than the number of positions in the file)
position0 = position0[0:numFrames]

# show positions
plt.figure(figsize=(10, 10))
plt.plot(position0[:, 0], position0[:, 1], 'o-')
plt.xlabel('pixel')
plt.ylabel('pixel')
plt.show()

# set propagator
propagatorType = 'Fraunhofer'

# export data
exportBool = True
if exportBool:
    hf = h5py.File(fileName+'.hdf5', 'w')
    hf.create_dataset('ptychogram', data=ptychogram, dtype='i')
    hf.create_dataset('encoder', data=encoder, dtype = 'f')
    # hf.create_dataset('position0', data=position0)
    hf.create_dataset('dxd', data=(dxd,), dtype = 'f')
    hf.create_dataset('Nd', data=(Nd,), dtype='i')
    hf.create_dataset('No', data=(No,), dtype='i')
    hf.create_dataset('Np', data=(Np,), dtype='i')
    hf.create_dataset('zo', data=(zo,), dtype='f')
    hf.create_dataset('wavelength', data=(wavelength,), dtype='f')
    hf.create_dataset('entrancePupilDiameter', data=(entrancePupilDiameter,), dtype='f')
    hf.close()
