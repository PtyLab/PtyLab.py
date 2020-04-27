import pickle
import numpy as np
import scipy.stats as st

def fft2c(array):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array),norm='ortho'))

def ifft2c(array):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array),norm='ortho'))

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def circ(x,y,D):
    """
    generate a circle on a 2D grid
    :param x: 2D array
    :param y: 2D array
    :param D: diameter 
    :return: a 2D array
    """
    circle = (x**2+y**2)<(D/2)**2
    return circle
