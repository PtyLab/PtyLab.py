import numpy as np

def GenerateNonUniformFermat(n, radius=1000, power=1):
    """
    generate spiral patterns
    :param n: number of points generated
    :param radius: radius in micrometer
    :param power = 1 is standard Fermat, power>1 yields more points towards the center of grid
    :return:
    R: row
    C: column
    """
    # golden ratio
    r = np.sqrt(np.arange(0, n)/n)
    theta0 = 137.508/180*np.pi
    theta = np.arange(0, n)*theta0
    C = radius*r**power*np.cos(theta)
    R = radius*r**power*np.sin(theta)
    return R, C

def GenerateConcentricGrid(Nr, s, rend):
    """
    generate concentric circles
    :param Nr: number of circles (or shells)
    :param s: number of pixels between
    :param rend: end radius size (in pixel units)
    """
    dx = 1  # max Resolution (Schritt von einem zum anderen Pixel)
    rstart = dx
    r = np.linspace(rstart, rend, Nr)
    # determine number of positions on k'th shell
    nop = np.zeros(Nr, dtype=int)
    for k in np.arange(Nr):
        nop[k] = int(np.floor(2 * np.pi * r[k] / s))
    positions = np.zeros((sum(nop) + 1, 2))
    ind = 1
    for k in np.arange(1, Nr):
        dtheta = 2 * np.pi / nop[k]
        theta = np.arange(1, nop[k] + 1) * dtheta + 2 * np.pi / (k + 1)
        for l in np.arange(nop[k]):
            positions[ind, :] = r[k] * np.array([np.cos(theta[l]), np.sin(theta[l])])
            ind += 1
    positions = (np.floor(positions / dx)).astype(int)

    R = positions[:, 0]
    C = positions[:, 1]
    return R, C

def GenerateRasterGrid(n, ds, randomOffset = False, amplitude = 1):
    """
    function to generate raster grid.
    :param n: number of points per dimension
    :param ds:
    :return:
    R: row
    C: column
    """
    I, J = np.meshgrid(np.arange(n), np.arange(n))
    C = I.reshape(n ** 2) * ds
    R = J.reshape(n ** 2) * ds

    if np.mod(n, 2) == 0:
        C = C - n * ds / 2
        R = R - n * ds / 2
    else:
        C = C - (n - 1) * ds / 2
        R = R - (n - 1) * ds / 2

    if randomOffset:
        C = C + np.round(amplitude * (-1 + 2 * np.random.rand(C.shape)))
        R = R + np.round(amplitude * (-1 + 2 * np.random.rand(R.shape)))

    R = np.round(R - np.mean(R)).astype(int)
    C = np.round(C - np.mean(C)).astype(int)

    return R, C






