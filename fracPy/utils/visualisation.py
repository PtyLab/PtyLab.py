# This file contains utilities required for monitors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO: copy-pase all relevant functions from Matlab implementation

def CoherencePlot():
    raise NotImplementedError

def grs2rgb():
    raise NotImplementedError

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb (5 times faster than colorsys).
    https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
    h,s should be a numpy arrays with values between 0.0 and 1.0
    v should be a numpy array with values between 0.0 and 255.0
    :param hsv: np.ndarray of shape (x,y,3)
    :return: hsv_to_rgb returns an array of uints between 0 and 255.
    """
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def complex_to_rgb(u):
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    h = np.angle(u)
    h = (h + np.pi) / (2 * np.pi)
    # saturation  (ones)
    s = np.ones_like(h)
    # value (normalize brightness to 8-bit)
    v = abs(u)
    v = v / (np.max(v) + np.finfo(float).eps) * 2 ** 8

    hsv = np.dstack([h, s, v])
    rgb = hsv_to_rgb(hsv)
    return rgb



def complex_plot(rgb, ax=None, pixelSize=1, axisUnit = 'mm'):
    """
    Plot a 2D complex np.ndarray, hue for phase, brightness for amplitude. Prepare a 2D complex array by using
    complex_to_rgb function.
    :param rgb: A 2D complex np.ndarray
    :param ax: Optional axis to plot in
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :param str axisUnit: Options: 'm', 'cm', 'mm', 'um'
    :return: An hsv plot
    """

    if not ax:
        fig, ax = plt.subplots()
    unitRatio = {'m': 1, 'cm':1e2, 'mm': 1e3, 'um': 1e6}
    pixelSize = pixelSize*unitRatio[axisUnit]
    extent = [0, pixelSize * rgb.shape[0], 0, pixelSize * rgb.shape[1]]
    im = ax.imshow(rgb, extent=extent)
    ax.set_ylabel(axisUnit)
    ax.set_xlabel(axisUnit)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, cax=cax, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])

    return im



# def hsvmodeplot(P,ax=None ,normalize = True, pixelSize =1):
#     """
#     Place multi complex images in a squre grid and use hsvplot to display
#     :param P: A complex np.ndarray
#     :param normalize: normalize each mode individually
#     :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
#     :return: A big array with flattened modes
#     """
#     if P.ndim>2 and P.shape[0]>1:
#         S = P.shape[0]
#         s = math.floor(np.sqrt(S))
#         if normalize:
#             maxs = np.max(P, axis=(1, 2))
#             P = (P.T/maxs).T
#
#         P = P.reshape((s, s) + P.shape[1:]).transpose(
#             (1, 2, 0, 3) + tuple(range(4, P.ndim + 1)))
#         P = P.reshape(
#             (s * P.shape[1], s * P.shape[3]) + P.shape[4:])
#     else:
#         P = np.squeeze(P)
#     im = hsvplot(P, ax = ax, pixelSize = pixelSize)
#
#     return im

def modeTile(P,normalize = True):
    """
    Tile 3D data into a single 2D array
    :param P: A complex np.ndarray
    :param normalize: normalize each mode individually
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :return: A big array with flattened modes
    """
    if P.ndim>2 and P.shape[0]>1:
        S = P.shape[0]
        s = math.floor(np.sqrt(S))
        if normalize:
            maxs = np.max(P, axis=(1, 2))
            P = (P.T/maxs).T

        P = P.reshape((s, s) + P.shape[1:]).transpose(
            (1, 2, 0, 3) + tuple(range(4, P.ndim + 1)))
        P = P.reshape(
            (s * P.shape[1], s * P.shape[3]) + P.shape[4:])
    else:
        P = np.squeeze(P)
    return P

