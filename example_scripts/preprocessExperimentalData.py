# This script contains some example of preprocessing an dataset into a hdf5 file that can be read into the
# fracPy dataset.
import numpy as np


# wavelength
wavelength = 450e-9
# binning
biningFactor = 4
# padding for superresolution
padFactor = 1
# set magnification if any objective lens is used
magfinication = 1
# object detector distance  (initial guess)
zo = 19.23e-3
# camera
N = 1456
M = 1456
dxd = 4.54e-6 * biningFactor / magfinication
backgroundOffset = 30

# number of frames is calculated automatically
numFrames = len

# read background
dark = imread('background.tif')


# read empty beam (if available)


# binning
ptychogram = np.zeros((N/biningFactor*padFactor, N/biningFactor*padFactor,
                       numFrames),dtype='complex64')

for k in range(numFrames)