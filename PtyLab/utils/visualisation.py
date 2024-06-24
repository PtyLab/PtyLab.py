# This file contains utilities required for Monitor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule, isGpuArray
from matplotlib.colors import LinearSegmentedColormap

try:
    import pyqtgraph as pg
except ModuleNotFoundError:
    print("Pyqtgraph not available")


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """
    Convert a 3D hsv np.ndarray to rgb (5 times faster than colorsys).
    https://stackoverflow.com/questions/27041559/rgb-to-hsv-python-change-hue-continuously
    h,s should be a numpy arrays with values between 0.0 and 1.0
    v should be a numpy array with values between 0.0 and 255.0
    :param hsv: np.ndarray of shape (x,y,3)
    :return: hsv2rgb returns an array of uints between 0 and 255.
    """
    xp = getArrayModule(hsv)
    rgb = xp.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype("uint8")
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5, i == i]
    rgb[..., 0] = xp.select(conditions, [v, q, p, p, t, v, v])  # , default=v)
    rgb[..., 1] = xp.select(conditions, [v, v, v, q, p, p, t])  # , default=t)
    rgb[..., 2] = xp.select(conditions, [v, p, t, v, v, q, p])  # , default=p)
    return rgb.astype("uint8")


def complex2rgb(u, amplitudeScalingFactor=1, force_numpy=True, center_phase=False):
    """
    Preparation function for a complex plot, converting a 2D complex array into an rgb array
    :param u: a 2D complex array
    :return: an rgb array for complex plot
    """
    # hue (normalize angle)
    # if u is on the GPU, remove it as we can toss it now.
    xp = getArrayModule(u)
    # u = asNumpyArray(u)
    if center_phase:
        N = u.shape[-1]
        phexp = xp.sum(u[...,N//3:2*N//3,N//3:2*N//3], axis=(-2,-1))
        u = u * phexp.conj() / (abs(phexp) + 1e-9)
    h = xp.angle(u)
    h = (h + np.pi) / (2 * np.pi)
    # saturation  (ones)
    s = xp.ones_like(h)
    # value (normalize brightness to 8-bit)
    v = xp.abs(u)
    if amplitudeScalingFactor == "2sigma":
        ASF = v.mean() + 2 * np.std(v)
        ASF = ASF / v.max()
    elif amplitudeScalingFactor is None:
        ASF = 1./v.max()
        amplitudeScalingFactor = ASF
    else:
        ASF = amplitudeScalingFactor
    
    if ASF != 1 and amplitudeScalingFactor != "2sigma":
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    v = v / (xp.max(v) + xp.finfo(float).eps) * (2**8 - 1)

    hsv = xp.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    if isGpuArray(rgb) and force_numpy:
        rgb = rgb.get()
    return rgb


def complex2rgb_vectorized(probe, **kwargs):
    """Turn complex image into rgb for every line.

    The individual images are all autoscaled, so you cannot compare them.
    """
    xp = getArrayModule(probe)
    original_shape = probe.shape
    probe = probe.reshape(-1, *probe.shape[-2:])
    probe_rgb = xp.array([complex2rgb(p, force_numpy=False, **kwargs) for p in probe])
    probe_rgb = probe_rgb.reshape(original_shape + (3,))
    return probe_rgb


def complexPlot(rgb, ax=None, pixelSize=1, axisUnit="pixel"):
    """
    Plot a 2D complex plot (hue for phase, brightness for amplitude). Input array need to be prepared by using
    the complex2rgb function.
    :param rgb: a rgb array that is converted from a 2D complex np.ndarray by using complex2rgb
    :param ax: Optional axis to plot in
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :param str axisUnit: Options: default 'pixel', 'm', 'cm', 'mm', 'um'
    :return: An hsv plot
    """

    if not ax:
        fig, ax = plt.subplots()
    unitRatio = {"pixel": 1, "m": 1, "cm": 1e2, "mm": 1e3, "um": 1e6}
    pixelSize = pixelSize * unitRatio[axisUnit]
    extent = [0, pixelSize * rgb.shape[1], pixelSize * rgb.shape[0], 0]

    im = ax.imshow(rgb, extent=extent, interpolation=None)
    ax.set_ylabel(axisUnit)
    ax.set_xlabel(axisUnit)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, ax=ax, cax=cax, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels(["$-\pi$", "0", "$\pi$"])
    return im


def modeTile(P, normalize=True):
    """
    Tile 3D data into a single 2D array
    :param P: A complex np.ndarray
    :param normalize: normalize each mode individually
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :return: A big array with flattened modes
    """
    if P.ndim == 3 and P.shape[0] > 1:
        if normalize:
            maxs = np.max(abs(P), axis=(-1, -2)) + 1e-6
            P = (P.T / maxs).T
        S = P.shape[0]
        s = math.ceil(np.sqrt(S))
        if s > np.sqrt(S):
            P = np.pad(P, ((0, s**2 - S), (0, 0), (0, 0)), "constant")
        P = P[: s**2, ...]
        P = P.reshape((s, s) + P.shape[1:]).transpose(
            (1, 2, 0, 3) + tuple(range(4, P.ndim + 1))
        )
        P = P.reshape((s * P.shape[1], s * P.shape[3]) + P.shape[4:])
    elif P.ndim == 4 and P.shape[0] > 1:
        if normalize:
            maxs = np.max(abs(P), axis=(-1, -2)) + 1e-6
            P = (P.T / maxs.T).T
        P = np.swapaxes(P, 1, 2).reshape(
            P.shape[0] * P.shape[2], P.shape[1] * P.shape[3]
        )
    else:
        P = np.squeeze(P)
    return P


def hsvplot(u, ax=None, pixelSize=1, axisUnit="pixel", amplitudeScalingFactor=1):
    """
    perform complex plot
    :param ax
    :param pixelSize, default 1
    :param axisUnit, default 'pixel', options: 'm', 'cm', 'mm', 'um'
    return: a complex plot
    """
    u = np.squeeze(asNumpyArray(u))
    rgb = complex2rgb(u, amplitudeScalingFactor=amplitudeScalingFactor)
    complexPlot(rgb, ax, pixelSize, axisUnit)


def hsvmodeplot(
    P, ax=None, normalize=True, pixelSize=1, axisUnit="pixel", amplitudeScalingFactor=1
):
    """
    Place multi complex images in a square grid and use hsvplot to display
    :param P: A complex np.ndarray
    :param normalize: normalize each mode individually
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :return: a tiled complex plot
    """

    Q = modeTile(np.squeeze(asNumpyArray(P)), normalize=normalize)
    hsvplot(
        Q,
        ax=ax,
        pixelSize=pixelSize,
        axisUnit=axisUnit,
        amplitudeScalingFactor=amplitudeScalingFactor,
    )


def absplot(
    u, ax=None, pixelSize=1, axisUnit="pixel", amplitudeScalingFactor=1, cmap="gray"
):
    U = np.abs(asNumpyArray(u))
    if not ax:
        fig, ax = plt.subplots()
    unitRatio = {"pixel": 1, "m": 1, "cm": 1e2, "mm": 1e3, "um": 1e6}
    pixelSize = pixelSize * unitRatio[axisUnit]
    extent = [0, pixelSize * U.shape[1], pixelSize * U.shape[0], 0]

    if amplitudeScalingFactor != 1:
        U[U > amplitudeScalingFactor * np.max(U)] = amplitudeScalingFactor * np.max(U)
    im = ax.imshow(U, extent=extent, interpolation=None, cmap=cmap)
    ax.set_ylabel(axisUnit)
    ax.set_xlabel(axisUnit)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    norm = mpl.colors.Normalize(vmin=0, vmax=amplitudeScalingFactor)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = plt.colorbar(
        scalar_mappable,
        ax=ax,
        cax=cax,
        ticks=[0, amplitudeScalingFactor / 2, amplitudeScalingFactor],
    )
    cbar.ax.set_yticklabels(
        ["0", str(amplitudeScalingFactor / 2), str(amplitudeScalingFactor)]
    )


def absmodeplot(
    P, ax=None, normalize=True, pixelSize=1, axisUnit="pixel", amplitudeScalingFactor=1
):
    Q = modeTile(abs(P), normalize=normalize)
    absplot(Q, ax=ax, pixelSize=pixelSize, axisUnit=axisUnit)


def setColorMap():
    """
    create the colormap for diffraction data (the same as matlab)
    return: customized matplotlib colormap
    """
    colors = [
        (1, 1, 1),
        (0, 0.0875, 1),
        (0, 0.4928, 1),
        (0, 1, 0),
        (1, 0.6614, 0),
        (1, 0.4384, 0),
        (0.8361, 0, 0),
        (0.6505, 0, 0),
        (0.4882, 0, 0),
    ]

    n = 255  # Discretizes the interpolation into n bins
    cm = LinearSegmentedColormap.from_list("cmap", colors, n)
    return cm


def show3Dslider(A, colormap="diffraction"):
    """
    show a 3D plot with a slider using pyqtgraph.
    :param A: a 3D array
    :param colormap: matplotlib colormap, default, customized colormap for plotting diffraction data
    return: a pyqtgraph plot
    """
    print(A.min(), A.max())
    app = pg.mkQApp()
    imv = pg.ImageView(view=pg.PlotItem())
    imv.setWindowTitle("Close to proceed")

    imv.setImage(A)

    # choose colormap from matplotlib colormaps
    if colormap == "diffraction":
        cmap = setColorMap()
    else:
        cmap = mpl.cm.get_cmap(colormap)

    # set the colormap
    positions = np.linspace(0, 1, cmap.N)
    colors = [(np.array(cmap(i)[:-1]) * 255).astype("int") for i in positions]
    imv.setColorMap(pg.ColorMap(pos=positions, color=colors))
    imv.show()
    app.exec_()
