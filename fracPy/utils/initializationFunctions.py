import numpy as np
from fracPy.utils.utils import circ, fft2c, ifft2c
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

def initialProbeOrObject(shape, type_of_init, data):
    """
    Implements line 148 of initialParams.m
    :param shape:
    :param type_of_init:
    :return:
    """
    if type(type_of_init) is np.ndarray: # it has already been implemented
        return type_of_init
    
    if type_of_init not in ['circ', 'rand', 'gaussian', 'ones', 'fpm', 'fpm_circ']:
        raise NotImplementedError()
        
    if type_of_init == 'ones':
        # NB this is how it's implemented, there's a bit of noise
        shape = np.asarray(shape)
        return np.ones(shape) + 0.001 * np.random.rand(*shape)
    
    if type_of_init == 'rand':
        # TODO: improve
        obj = np.exp(1j * 2*np.pi * (np.random.rand(*shape)-1/2))
        for idx in range(shape[0]):
            obj[idx,:,:] =  gaussian_filter(np.abs(obj[idx,:,:]), 3) * np.exp(1j * gaussian_filter(np.angle(obj[idx,:,:]), 3))
        return obj

    if type_of_init == 'circ':
        # get the circular boundary constraints
        try:
            shape = np.asarray(shape)
            if 'aperture' in dir(data):
                pupil = data.aperture.copy()
            else:
                pupil = circ(data.Xp, data.Yp, data.entrancePupilDiameter)
            # return the initial low-pass filtered array
            return np.ones(shape) * pupil + 0.001 * np.random.rand(*shape)
        
        except AttributeError as e:
            raise AttributeError(e, 'probe/aperture/entrancePupilDiameter was not defined')

    if type_of_init == 'fpm':
        # upsample the ptychogram using fourier space padding to create an object
        # estimate
        # padSize = np.int((data.No - data.Np) / 2)
        # upsampledObjectSpectrum = np.pad(fft2c(np.mean(data.ptychogram,0)), (padSize,padSize),'constant')     
        # # return the upsampled image
        # upsampledObjectSpectrum = fft2c(np.abs(ifft2c(upsampledObjectSpectrum)))
        # return np.ones(shape)*upsampledObjectSpectrum
        return np.ones(shape)*ifft2c(rescale(np.mean(data.ptychogram,0), data.No/data.Np))
    
    if type_of_init == 'fpm_circ':
        # get the circular boundary constraints for FPM. Due to a physical aperture
        # stop it must not have any random noise added
        try:
            shape = np.asarray(shape)
            if 'aperture' in dir(data):
                pupil = data.aperture.copy()
            else:
                pupil = circ(data.Xp, data.Yp, data.entrancePupilDiameter)
            # return the initial low-pass filtered array
            return np.ones(shape) * pupil
        
        except AttributeError as e:
            raise AttributeError(e, 'probe/aperture/entrancePupilDiameter was not defined')