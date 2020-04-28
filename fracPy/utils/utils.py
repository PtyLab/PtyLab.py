import pickle
import numpy as np
import scipy.stats as st


def fft2c(array):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array),norm='ortho'))
    # return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(array,axes=(1,2)),norm='ortho',axes=(1,2) ),axes=(1,2))


def ifft2c(array):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array),norm='ortho'))
    # return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(array,axes=(1,2)),norm='ortho',axes=(1,2) ),axes=(1,2)) 

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
