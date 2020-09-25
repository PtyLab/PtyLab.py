import numpy as np

def GenerateNonUniformFermat(n, radius=1000, power=1):
    """
    generate spiral patterns
    power = 1 is standard Fermat, power>1 yields more points towards the center of grid
    n: number of points generated
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