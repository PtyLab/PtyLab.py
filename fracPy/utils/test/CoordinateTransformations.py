import time
import numpy as np


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

def utoX(u: np.ndarray, v: np.ndarray, z,wavelength,xref,yref):
    """
    Maps spatial frequencies of the exit wave(u,v) to detector coordinates(x,y), the inverse of xtoU
    :param u: spatial frequencies associated with the x coordinates of the exit wave
    :param v:spatial frequencies associated with the y coordinates of the exit wave
    :param wavelength:
    :return: detector space grid(x,y)
    :rtype:
    """
    x = (u * wavelength * z) * np.sqrt((1. / (1 - (wavelength ** 2 * (u ** 2+v ** 2)))))
    y = (v * wavelength * z) * np.sqrt((1. / (1 - (wavelength ** 2 * (u ** 2+v ** 2)))))
    return x, y



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
    theta = toRadians(theta)

    ro = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    v = y / (wavelength * ro)
    u = (x * np.cos(theta) - np.sin(theta) * (ro - z)) / (wavelength * ro)

    return u, v


def tiltUtoX(u: np.ndarray, v: np.ndarray, z, wavelength, theta):
    """
    Warning needs 64 bit precision in u and v near 45 degrees(and potentially other places)

    Maps from tilted sample-plane spatial frequencies(of the exit wave) to corresponding detector plane coordinates,
    when dealing with a non co-linear sample-detector geometry

    :param u: spatial frequencies associated with x in sample coordinates,
    :param v: spatial frequencies associated with y in sample coordinates,
    :param wavelength:illumination wavelength
    :param theta: tilt angle between sample plane and detector plane in degrees
    :return:
    :rtype: x,y(the detector coordinates associated with the input spatial frequencies, after transform inversion)
    """


    # for derivation see ~placeholder for pdf
    theta = toRadians(theta)
    a = 1. + (((u / v) + np.sin(theta) / (v * wavelength)) / np.cos(theta)) ** 2 - (v * wavelength) ** (-2)
    b = -(2. * np.sin(theta) * z / (v * np.cos(theta) ** 2)) * (
            u + np.sin(theta) / wavelength)
    c = (1 + np.tan(theta) ** 2) * z ** 2
    y = (-b - np.sign(v) * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    # y is not invertible with this quadratic equation at v=0 as 0/0 is undefined, but y=lambda*r0*v, so y=0
    y = np.where(v == 0, 0, y)

    # at a=0 we again gets something that is not defined, furthermore, the numerical accuracy due to
    # rounding
    # error s becomes problematic when a=very small, so we replace by the linear equation(bx=-cy) for those cases.
    y = np.where(abs(a) > 1.e-6, y, -c / b)

    # to determine the sign of the root for x we  calculate the u spatial frequencies where x=0, this marks a uxo(y)
    # line. all coordinates that have a u(x, y) that is larger then uxo(y) are positive.
    uxo = np.sin(theta) * (z - np.sqrt(z ** 2 + y ** 2)) / (wavelength * np.sqrt(z ** 2 + y ** 2))
    x = np.sign(u - uxo) * np.real(np.sqrt(((v * wavelength) ** (-2) - 1) * y ** 2 - z ** 2))
    # when y=0,v=0 this equation is not defined so we invert for y=0, and get another quadratic equation for x
    ax = (np.cos(2 * theta) - 2 * wavelength * u[abs(v) < 1e-6] * np.sin(theta) - (wavelength * u[abs(v) < 1e-6]) ** 2)
    bx = 2 * z * np.sin(theta) * np.cos(theta)
    cx = -(z ** 2 * ((wavelength * u[abs(v) < 1e-6]) * (2 * np.sin(theta) + (wavelength * u[abs(v) < 1e-6]))))

    x2 = (-bx + np.sqrt(bx ** 2 - 4 * ax * cx)) / (2 * ax)
    x2 = np.where(np.abs(ax) < 1.e-6, -cx / bx, x2)

    x = np.where(np.abs(v) < 1.e-6, x2, x)
    # x is unstable near x=0 in some cases, especially near 45 degrees(y has large rounding errors when a is small and
    # the expression for x is sensitive to rounding errors in y, as effectively your trying to infer x from the
    # departure of the linear relationship between y and v), the departure of linearity is not so large at low NA and
    # the numerically instability of y for small(but not zero) a hurts double.
    # however we calculated at which spatial frequencies x=0 before and replace a region where rounding errors are
    # relevant around that with zeros
    x = np.where(np.abs(u - uxo) < 1.e-6, 0, x)

    return x, y




def toRadians(theta):
    theta = theta * np.pi / 180.
    return theta

def two_norm(array):
    norm=np.sum(np.abs(array)**2)
    return norm
zo = 70.2e-3
theta = 45
dx = 1.476e-05
Nd = 676
wavelength = 700e-9

dxd = np.arange(-Nd / 2, Nd / 2 + 2) * dx

xd, yd = np.meshgrid(dxd, dxd)

begin = time.time()
uq, vq = xtoU(xd, yd, zo,  wavelength)
xq, yq = utoX(uq, vq, zo, wavelength,xd,yd)
end = time.time() - begin
norm=two_norm(yq-yd)

print(end)
outer_points = np.array([[xd[-1, -1], yd[-1, -1]], [xd[0, 0], yd[-1, -1]], [xd[-1, -1], yd[0, 0]], [xd[0, 0], yd[0,
                                                                                                                 0]]])