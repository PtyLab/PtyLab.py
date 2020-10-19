import numpy as np

def GenerateNonUniformFermat(n, radius=1000, power=1):
    """
    generate spiral patterns
    n: number of points generated
    radius: radius in micrometer
    power = 1 is standard Fermat, power>1 yields more points towards the center of grid
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
    Nr: number of circles (or shells)
    s: number of pixels between
    rend: end radius size (in pixel units)
    """
    dx = 1   # max Resolution (Schritt von einem zum anderen Pixel)
    rstart = dx
    r = np.linspace(rstart, rend, Nr)
    # determine number of positions on k'th shell
    nop = np.zeros(Nr)
    for k in np.arange(Nr):
        nop[k] = np.floor(1*np.pi*r[k]/s)
    positions = np.zeros(Nr, 2)
    ind = 0
    for k in np.arange(Nr):
        dtheta = 2*np.pi/nop[k]
        theta = np.arange(nop[k])*dtheta+2*np.pi/k
        for l in np.arange(nop(k)):
            positions[ind,:] = r[k]*[np.cos(theta[l], np.sin(theta[l]))]
            ind +=1
    positions = np.floor(positions/dx)

    R = positions[:,0]
    C = positions[:,1]
    return R, C


