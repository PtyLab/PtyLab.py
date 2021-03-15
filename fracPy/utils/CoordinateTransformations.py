import time
import numpy as np
import matplotlib.pyplot as plt

from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray

def prepareUVgrid(x, y, zo, wavelength):
    """
    Generate a Spatial frequency space grid used in the reconstruction, find the largest spatial frequencies
    associated with the detector grid and then create an equally sampled grid that includes those spatial
    frequencies, and sets the sample space pixel sizes according to that spatial frequency range.

    """

    Fdetx, Fdety = xtoU(x, y, zo,
                        wavelength)
    lowerbound = np.amin(Fdetx)
    upperbound = np.amax(Fdetx)
    df = Fdetx[338, 338] - Fdetx[338, 339]
    Ugrid = np.arange(lowerbound, upperbound, 676, dtype=np.float64)

    U, V = np.meshgrid(Ugrid, Ugrid, sparse=True)
    return U, V


def xtoU(x: np.ndarray, y: np.ndarray, z, wavelength):
    """
    Maps detector coordinates (x,y) to spatial frequencies of the exit wave(u,v) with a conversion that is more
    accurate then the
    as opposed to the more common approximation u=x/(lambda*z). Cost of having to use interpolation to get an evenly
    spaced (u,v) grid often outweighs te cost(as in the far field r~z), if you do so consider using the inverse
    transform(utoX), as interpolation in inverse mapping is better behaved.
    :param x: detector space coordinate
    :param y: detector space coordinate
    :param z: detector-sample distance
    :param wavelength:
    :return: u,v : spatial frequencies dual to the exit wave( object pixel size x=1/u, y=1/v)
    """
    ro = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    u = x / (wavelength * ro)
    v = y / (wavelength * ro)
    return u, v


def utoX(u: np.ndarray, v: np.ndarray, z, wavelength):
    """
    Maps spatial frequencies of the exit wave(u,v) to detector coordinates(x,y), the inverse of xtoU
    :param u: spatial frequencies associated with the x coordinates of the exit wave
    :param v:spatial frequencies associated with the y coordinates of the exit wave
    :param wavelength:
    :return: detector space grid(x,y)
    :rtype:
    """
    x = (u * wavelength * z) * np.sqrt((1. / (1 - (wavelength ** 2 * (u ** 2 + v ** 2)))))
    y = (v * wavelength * z) * np.sqrt((1. / (1 - (wavelength ** 2 * (u ** 2 + v ** 2)))))
    return x, y


def quickshow(array):
    plt.imshow(array)
    plt.colorbar()
    plt.show()


def xtoTiltU(x: np.ndarray, y: np.ndarray, z, wavelength, theta):
    """
    Maps detector plane coordinates to corresponding  tilted plane spatial frequency coordinates
    :param x: detector x coordinates
    :param y: detector y coordinates
    :param z: detector-sample distance
    :params wavelength: illumination wavelength
    :param theta: tilt angle between sample plane and detector plane in degrees
    :return: warped spatial frequencies u,v associated with x,y
    """
    module = getArrayModule(x)
    theta = toRadians(theta)

    ro = module.sqrt(x ** 2 + y ** 2 + z ** 2)
    v = y / (wavelength * ro)
    u = (x * module.cos(theta) - module.sin(theta) * (ro - z)) / (wavelength * ro)

    return u, v


def tiltUtoX(u: np.ndarray, v: np.ndarray, z, wavelength, theta, axis=0, output_list=True):
    """
    Warning needs 64 bit precision in u and v near 45 degrees(and potentially other places)

    Maps from tilted sample-plane spatial frequencies(of the exit wave) to corresponding detector plane coordinates,
    when dealing with a non co-linear sample-detector geometry


    :type output_list:
    :param u: spatial frequencies associated with x in sample coordinates(NxN grid),
    :param v: spatial frequencies associated with y in sample coordinates(NxN grid),
    :param wavelength:illumination wavelength
    :param theta: tilt angle between sample plane and detector plane in degrees
    :param axis: axis where the detector signal is warped and needs correction, defaults to the x-axis(u)
    :param output_list: if this is true this function will output a list(as used for the interpolation function used
    in aPIE
    :return:
    :rtype: x,y(the detector coordinates associated with the input spatial frequencies, after transform inversion)
    """

    module = getArrayModule(u)

    # for derivation see ~placeholder for pdf

    if axis == 0:
        uw = u
        unow = v
        # allow switching the tilt-axis, the documentation assumes the warped axis(uw) is u and the non warped axis(
        # unow) is v
    if axis == 1:
        uw = v
        unow = u
    theta = toRadians(theta)
    a = 1. + (((uw / unow) + module.sin(theta) / (unow * wavelength)) / module.cos(theta)) ** 2 - (unow * wavelength) ** \
        (-2)
    b = -(2. * module.sin(theta) * z / (unow * module.cos(theta) ** 2)) * (
            uw + module.sin(theta) / wavelength)
    c = (1 + module.tan(theta) ** 2) * z ** 2
    y = (-b - module.sign(unow) * module.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    # y is not invertible with this quadratic equation at v=0 as 0/0 is undefined, but y=lambda*r0*v, so y=0
    y = module.where(abs(unow) < 1.e-7, 0, y)

    # at a=0 we again gets something that is not defined, furthermore, the numerical accuracy due to
    # rounding
    # error s becomes problematic when a=very small, so we replace by the linear equation(bx=-cy) for those cases.
    y = module.where(abs(a) > 1.e-6, y, -c / b)

    # to determine the sign of the root for x we  calculate the u spatial frequencies where x=0, this marks a uxo(y)
    # line. all coordinates that have a u(x, y) that is larger then uxo(y) are positive.
    uxo = module.sin(theta) * (z - module.sqrt(z ** 2 + y ** 2)) / (wavelength * module.sqrt(z ** 2 + y ** 2))
    x = module.sign(uw - uxo) * module.sqrt(abs(((unow * wavelength) ** (-2) - 1) * y ** 2 - z ** 2))
    # x[(((unow * wavelength) ** (-2) - 1) * y ** 2 - z ** 2) < 0] = 0
    # when y=0,v=0 this equation is not defined so we invert for y=0, and get another quadratic equation for x
    ax = (module.cos(2 * theta) - 2 * wavelength * uw * module.sin(theta) - (wavelength * uw) ** 2)

    bx = 2 * z * module.sin(theta) * module.cos(theta)
    cx = -(z ** 2 * ((wavelength * uw) * (2 * module.sin(theta) + (wavelength * uw))))

    x2 = (-bx + module.sqrt(bx ** 2 - 4 * ax * cx)) / (2 * ax)
    x2 = module.where(module.abs(ax) < 1.e-5, -cx / bx, x2)

    x = module.where(module.abs(uw - uxo) < 1.e-5, 0, x)
    x = module.where(v == 0, x2, x)

    # x is unstable near x=0 in some cases, especially near 45 degrees(y has large rounding errors when a is small and
    # the expression for x is sensitive to rounding errors in y, as effectively your trying to infer x from the
    # departure of the linear relationship between y and v), the departure of linearity is not so large at low NA and
    # the numerically instability of y for small(but not zero) a hurts double.
    # however we calculated at which spatial frequencies x=0 before and replace a region where rounding errors are
    # relevant around that with zeros
    xuo = (np.sin(theta) * (y/ (wavelength * v))-z * np.sin(theta)) / np.cos(theta)
    x=np.where(u==0,xuo,x)
    x[338, 338] = 0

    if output_list == 1:
        x = module.ravel(x)
        y = module.ravel(y)
    if axis == 0:
        return x, y
    else:
        return y, x


def toRadians(theta):
    theta = theta * np.pi / 180.
    return theta


def toDegrees(theta):
    theta = theta * (np.pi / 180.) ** -1
    return theta


def two_norm(array):
    module = getArrayModule(array)
    norm = module.linalg.norm(array, ord=2)
    return norm


if __name__ == "__main__":
    zo = 71.3e-3
    theta = 44
    dx = 1.476e-05
    Nd = 676
    wavelength = 708.8e-9

    dxd = np.arange(-Nd / 2, Nd / 2) * dx

    xd, yd = np.meshgrid(dxd, dxd, sparse=True)
    utest, vtest = prepareUVgrid(xd, yd, zo, wavelength)
    begin = time.time()
    # uq, vq = xtoU(xd, yd, zo,  wavelength)
    # xq, yq = tiltUtoX_testing(utest, vtest, zo, wavelength, theta, utest,vtest)
    xq, yq = tiltUtoX(utest, vtest, zo, wavelength, theta, output_list=0)
    end = time.time()
    print(end - begin)
    refu, refv = xtoTiltU(xq, yq, zo, wavelength, theta)
    # xqp, yqp = tiltUtoX_testing(utest, vtest, zo, wavelength, theta)
    norm = two_norm(refu - utest)
    difu = abs(refv - utest)
    plt.plot(refv[338, :], difu[338, :])

    outer_points = np.array([[xd[-1, -1], yd[-1, -1]], [xd[0, 0], yd[-1, -1]], [xd[-1, -1], yd[0, 0]], [xd[0, 0], yd[0,
                                                                                                                     0]]])
