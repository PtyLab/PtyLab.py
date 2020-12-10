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


filePathForRead = r"\\sun.amolf.nl\eikema-witte\group-folder\Phasespace\ptychography\rawData\vortexCharacterization\Vortex_measurements_multiwavelength_700_850nm\20201029_080936_VocationalGuidance00001\AVT camera (GT3400)"
filePathForSave = r"D:\fracPy_Antonios\example_data"
# D:\Du\Workshop\fracmat\lenspaper4\AVT camera (GX1920)
# D:/fracmat/ptyLab/lenspaper4/AVT camera (GX1920)
os.chdir(filePathForRead)

fileName = 'Vortex_as_obj_700_850'
# wavelength
spectralDensity = np.linspace(700e-9, 850e-9, 4)
wavelength = min(spectralDensity)
# wavelength = 708.9e-9 #450e-9
# binning
binningFactor = 4
# padding for superresolution
padFactor = 1
# set magnification if any objective lens is used
magnification = 1
# object detector distance  (initial guess)
zo = 61.95e-3 #19.23e-3
# set detection geometry
# A: camera to closer side of stage (allows to bring camera close in transmission)
# B: camera to further side of stage (doesn't allow to bring close in transmission),
# or other way around in reflection
# C: objective + tube lens in transmission
measurementMode = 'A'
# camera
camera = 'GT'
if camera == 'GX':
    N = 1456
    M = 1456
    dxd = 4.54e-6 * binningFactor / magnification # effective detector pixel size is magnified by binning
    backgroundOffset = 100 # globally subtracted from raw data (diffraction intensities), play with this value
elif camera == 'Hamamatsu':
    N = 2**11
    M = 2**11
    dxd = 6.5e-6 * binningFactor / magnification
    backgroundOffset = 30
elif camera == 'Manta':
    N = 2**11
    M = 2**11
    backgroundOffset = 40
    dxd = 5.5e-6 * binningFactor / magnification
elif camera == 'GT':
    N = 3384
    M = 2704
    dxd = 3.69e-6 * binningFactor / magnification;
    backgroundOffset = 5.0

# number of frames is calculated automatically
framesList = glob.glob('*'+'.tif')
framesList.sort()
numFrames = len(framesList)-1

# read background
dark = imageio.imread('background.tif')

# read empty beam (if available)

# binning
ptychogram = np.zeros((numFrames, M//binningFactor*padFactor, M//binningFactor*padFactor), dtype=np.float32)

# read frames
pbar = tqdm.trange(numFrames, leave=True)
for k in pbar:
    # get file name
    pbar.set_description('reading frame' + framesList[k])
    temp = imageio.imread(framesList[k]).astype('float32')-dark-backgroundOffset
    temp[temp < 0] = 0  #todo check if data type is single
    # crop
    # temp = temp[N//2-M//2:M//2+N//2-1, N//2-M//2:M//2+N//2-1]
    temp = temp[:, 85:761]

    # binning
    binningFactor = 1
    temp = rescale(temp, 1/binningFactor, order=1) # order = 0 takes the nearest-neighbor
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
# dxp = dxo
# Np = Nd
# Lp = Np * dxp
# xp = np.arange(-Np//2, Np//2) * dxp
# Xp, Yp = np.meshgrid(xp, xp)

# get positions
# get file name (this assumes there is only one text file in the raw data folder)
positionFileName = glob.glob('*'+'.txt')[0]

# take raw data positions
T = np.genfromtxt(positionFileName, delimiter=' ', skip_header=2)
# convert to micrometer
encoder = (T-T[0]) * 1e-6
# encoder = (T-T[0]) * 1e-6 / magfinication
# convert into pixels
# position0 = np.round(encoder / dxo)
# # center within object grid
# position0 = position0 + No // 2 - Np // 2
# # take only the frames needed (if numFrames smaller than the number of positions in the file)
# position0 = position0[0:numFrames]

# show positions
plt.figure(figsize=(5, 5))
plt.plot(encoder[:, 1]* 1e6, encoder[:, 0]* 1e6, 'o-')
plt.xlabel('(um))')
plt.ylabel('(um))')
plt.show()

# set propagator
# propagatorType = 'Fraunhofer'

# export data
exportBool = True

if exportBool:
    os.chdir(filePathForSave)
    hf = h5py.File(fileName+'.hdf5', 'w')
    hf.create_dataset('ptychogram', data=ptychogram, dtype='f')
    hf.create_dataset('encoder', data=encoder, dtype='f')
    hf.create_dataset('dxd', data=(dxd,), dtype='f')
    hf.create_dataset('Nd', data=(Nd,), dtype='i')
    hf.create_dataset('zo', data=(zo,), dtype='f')
    hf.create_dataset('spectralDensity', data=(spectralDensity,), dtype='f')
    hf.create_dataset('wavelength', data=(wavelength,), dtype='f')
    hf.create_dataset('entrancePupilDiameter', data=(entrancePupilDiameter,), dtype='f')
    hf.close()
    print('An hd5f file has been saved')
