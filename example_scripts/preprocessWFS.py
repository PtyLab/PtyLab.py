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
from fracPy.utils.utils import fraccircshift, posit


filePathForRead = r"D:\Du\Workshop\fracmat\lenspaper4\AVT camera (GX1920)"
filePathForRead =r"\\sun\eikema-witte\project-folder\XUV_lensless_imaging\backups\two-pulses\ARCNL\2020_11_04_ptycho_donut_Concentric_28oct_500micronFOV_50micron_stepsize"
filePathForSave = r"D:\Du\Workshop\fracpy\example_data"
# D:\Du\Workshop\fracmat\lenspaper4\AVT camera (GX1920)
# D:/fracmat/ptyLab/lenspaper4/AVT camera (GX1920)
os.chdir(filePathForRead)

fileName = 'WFS_fundamental'
# spectral density
spectralDensity = 762.2e-9
spectralDensity = 850e-9/np.arange(19, 35, 2)
# wavelength
wavelength = min(spectralDensity)
# binning
binningFactor = 1
# set magnification if any objective lens is used
magfinication = 1
# object detector distance  (initial guess)
zo = 192.0e-3
# HHG setup
cameraPixelSize = 13.5e-6
# number of pixels in raw data
P = 2048
# pixel in cropped data
N = 2**10  # NIR
# N = 2**9 # EUV
# dark/readout offset
backgroundOffset = 60

# object detector distance
zo = 192.0e-3 # object-detector distance


## set experimental specifications
# detector coordinates
dxd = cameraPixelSize*binningFactor # effective detector pixel size is magnified by binning
Nd = N                             # number of detector pixels

## get positions
# get file name (this assumes there is only one text file in the raw data folder)
positionFileName = glob.glob('*'+'.txt')[0]

# take raw data positions
T = np.genfromtxt(positionFileName, delimiter=' ', skip_header=1)  # HHG data, skip_header = 1, NIR data skip_hearder = 2
# match the scan grid with the ptychogram
T[:, 1] = -T[:, 1]
# convert to micrometer
encoder = (T-T[0]) * 1e-6
# convert into pixels
detectorShifts = T * 1e-6 / dxd
# show positions
plt.figure(figsize=(5, 5))
plt.plot(encoder[:, 1] * 1e6, encoder[:, 0] * 1e6, 'o-')
plt.xlabel('(um))')
plt.ylabel('(um))')
plt.show(block=False)

## read data and correct for darks
# number of frames is calculated automatically
framesList = glob.glob('*'+'.tif')
framesList.sort()
numFrames = len(framesList)-1

# read background
dark = imageio.imread('background.tif')

# binning
ptychogram = np.zeros((numFrames, P//binningFactor, P//binningFactor), dtype=np.float32)

# read frames
pbar = tqdm.trange(numFrames, leave=True)
for k in pbar:
    # get file name
    pbar.set_description('reading frame' + framesList[k])
    I = posit(imageio.imread(framesList[k]).astype('float32')-dark-backgroundOffset)
    ptychogram[k] = fraccircshift(I, -detectorShifts[k])

## crop data if necessary
if N < P:
    # center             
    x = np.arange(P)
    [X,Y] = np.meshgrid(x, x)
    # ptychogram_forCenter = ptychogram
    ptychogram_forCenter = posit(ptychogram-200)
    ptychogram_sum = np.sum(ptychogram_forCenter, axis=0)
    ptychogram_sum = ptychogram_sum/np.sum(ptychogram_sum)
    rowCenter = int(np.round(np.sum(ptychogram_sum*Y)))
    colCenter = int(np.round(np.sum(ptychogram_sum*X)))
    ptychogram = ptychogram[:,rowCenter-N//2:rowCenter+N//2, colCenter-N//2:colCenter+N//2]

    
# set experimental specifications:
entrancePupilDiameter = 1000e-6


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
    hf.create_dataset('wavelength', data=(wavelength,), dtype='f')
    hf.create_dataset('entrancePupilDiameter', data=(entrancePupilDiameter,), dtype='f')
    hf.close()
    print('An hd5f file has been saved')
