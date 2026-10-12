# This file contains utilities required for Monitor
import functools
import logging
import math
import os
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule, isGpuArray

logger = logging.getLogger(__name__)


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
        phexp = xp.sum(u[..., N // 3 : 2 * N // 3, N // 3 : 2 * N // 3], axis=(-2, -1))
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
        ASF = 1.0 / v.max()
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


# Axis units for the image plots. The reciprocal ones are for Fourier-space
# quantities such as the FPM pupil, whose axes are spatial frequencies.
unitRatio = {
    "pixel": 1,
    "m": 1,
    "cm": 1e2,
    "mm": 1e3,
    "um": 1e6,
    "1/m": 1,
    "1/mm": 1e-3,
    "1/um": 1e-6,
}


def plotExtent(pixelSize, axisUnit, shape):
    """
    Extent for imshow, expressed in axisUnit.

    Real-space axes run from zero, as they always have. Reciprocal axes are
    centred on zero frequency instead, which is where the pupil sits.

    :param pixelSize: sample spacing of the array, in SI units
    :param str axisUnit: any key of unitRatio
    :param shape: shape of the array that is plotted
    :return: [left, right, bottom, top] for imshow
    """
    step = pixelSize * unitRatio[axisUnit]
    width, height = step * shape[1], step * shape[0]
    if axisUnit.startswith("1/"):
        return [-width / 2, width / 2, height / 2, -height / 2]
    return [0, width, height, 0]


def complexPlot(rgb, ax=None, pixelSize=1, axisUnit="pixel"):
    """
    Plot a 2D complex plot (hue for phase, brightness for amplitude). Input array need to be prepared by using
    the complex2rgb function.
    :param rgb: a rgb array that is converted from a 2D complex np.ndarray by using complex2rgb
    :param ax: Optional axis to plot in
    :param pixelSize: pixelSize in x and y, to display the physical dimension of the plot
    :param str axisUnit: Options: default 'pixel', 'm', 'cm', 'mm', 'um', and the
        reciprocal '1/m', '1/mm', '1/um' for Fourier-space quantities
    :return: An hsv plot
    """

    if not ax:
        fig, ax = plt.subplots()
    extent = plotExtent(pixelSize, axisUnit, rgb.shape)

    im = ax.imshow(rgb, extent=extent, interpolation=None)
    ax.set_ylabel(axisUnit)
    ax.set_xlabel(axisUnit)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hsv)
    scalar_mappable.set_array([])
    # colorbar on the axes' own figure, not pyplot's current one
    cbar = ax.figure.colorbar(
        scalar_mappable, ax=ax, cax=cax, ticks=[-np.pi, 0, np.pi]
    )
    cbar.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
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
    extent = plotExtent(pixelSize, axisUnit, U.shape)

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


def _is_notebook():
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def _has_display():
    """Return True when a Qt window can be shown.

    On Linux (e.g. a GPU server reached over ssh) Qt needs an X11 or Wayland
    display; without one it aborts the whole process from C++, which Python
    cannot catch, so this has to be checked before Qt starts. Setting
    `QT_QPA_PLATFORM` explicitly (e.g. `offscreen`) is respected as an override.
    """
    if not sys.platform.startswith("linux"):
        return True
    return any(
        os.environ.get(v) for v in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")
    )


@functools.lru_cache(maxsize=1)
def _qt_start_error():
    """Return None if a QApplication can start here, else Qt's error message.

    A display being set is not enough on Linux: e.g. over `ssh -X` Qt >= 6.5 also
    needs the system library libxcb-cursor0, and when anything like that is
    missing Qt aborts the process from C++ (uncatchable from Python). So start a
    throwaway QApplication in a subprocess first. Cached: checked once per session.
    """
    if not sys.platform.startswith("linux"):
        return None
    code = "from pyqtgraph.Qt import QtWidgets; QtWidgets.QApplication([])"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)
    if result.returncode == 0:
        return None
    # Qt's first stderr line names the cause, e.g. "... libxcb-cursor0 is needed ..."
    lines = [l for l in result.stderr.splitlines() if l.strip()]
    return lines[0] if lines else f"exit code {result.returncode}"


def _show3Dslider_matplotlib(A, cmap):
    """Fallback 3D viewer using matplotlib's own Slider widget (no Qt needed)."""
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(6, 6.5))
    # leave room under the image for the slider axis
    fig.subplots_adjust(bottom=0.15)
    im = ax.imshow(A[0], cmap=cmap, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    slider_ax = fig.add_axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(slider_ax, "Frame", 0, A.shape[0] - 1, valinit=0, valstep=1)

    def update_frame(value):
        im.set_data(A[int(value)])
        fig.canvas.draw_idle()

    slider.on_changed(update_frame)
    # the canvas holds this lambda strongly, which keeps the slider alive while the
    # figure is open (with a non-blocking plt.show it would otherwise be collected)
    fig.canvas.mpl_connect("close_event", lambda _: slider.disconnect_events())
    plt.show()


def show3Dslider(A, colormap="diffraction"):
    """
    show a 3D plot with a slider.

    In a Jupyter notebook an inline ipywidgets slider is used.
    In a script the interactive pyqtgraph viewer is used. If pyqtgraph is not
    importable or there is no display (e.g. ssh without X11 forwarding), it
    falls back to a matplotlib slider.

    :param A: a 3D array
    :param colormap: matplotlib colormap, default, customized colormap for plotting diffraction data
    return: a pyqtgraph plot
    """
    print(A.min(), A.max())

    # resolve colormap once — matplotlib LinearSegmentedColormap works for both paths
    if colormap == "diffraction":
        cmap = setColorMap()
    else:
        cmap = mpl.colormaps[colormap]

    if _is_notebook():
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display

        # Create output widget for displaying figure
        out = widgets.Output()

        def update_frame(change):
            # Create new figure for each frame update
            # Handle both old API (event dict) and new API (direct value)
            if isinstance(change, dict):
                frame = int(change["new"])
            else:
                frame = int(change)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(A[frame], cmap=cmap, origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            plt.tight_layout()

            with out:
                out.clear_output(wait=True)
                display(fig)
                plt.close(fig)

        # Create slider widget
        slider = widgets.IntSlider(
            min=0, max=A.shape[0] - 1, step=1, value=0, description="Frame"
        )

        # Link slider to update function
        slider.observe(update_frame, names="value")

        # Display slider and output
        display(widgets.VBox([slider, out]))

        # Initial display
        update_frame(0)
    else:
        if not _has_display():
            logger.warning(
                "No display found (DISPLAY/WAYLAND_DISPLAY unset); using the "
                "matplotlib viewer instead of pyqtgraph."
            )
            _show3Dslider_matplotlib(A, cmap)
            return
        try:
            # optional `gui` extra; pulls in the Qt binding (PySide6 by default,
            # any binding pyqtgraph supports works)
            import pyqtgraph as pg
        except ImportError as e:
            logger.warning(
                "pyqtgraph/Qt not available (%s); using the matplotlib viewer. For the "
                'interactive GUI, install PtyLab with `pip install "ptylab[gui]"`.',
                e,
            )
            _show3Dslider_matplotlib(A, cmap)
            return
        # an existing QApplication proves Qt works; otherwise probe before starting one
        if pg.Qt.QtWidgets.QApplication.instance() is None:
            error = _qt_start_error()
            if error is not None:
                logger.warning(
                    "Qt cannot start (%s); using the matplotlib viewer. On Linux, "
                    "`sudo apt install libxcb-cursor0` usually fixes this.",
                    error,
                )
                _show3Dslider_matplotlib(A, cmap)
                return
        app = pg.mkQApp()
        imv = pg.ImageView(view=pg.PlotItem())
        imv.setWindowTitle("Close to proceed")
        imv.setImage(A)

        # set the colormap
        positions = np.linspace(0, 1, cmap.N)
        colors = [(np.array(cmap(i)[:-1]) * 255).astype("int") for i in positions]
        imv.setColorMap(pg.ColorMap(pos=positions, color=colors))
        imv.show()
        # exec() works on every Qt binding; exec_() is a deprecated PyQt5-era alias
        app.exec()
