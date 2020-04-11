import pickle



def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def circ(x,y,D)
    """
    generate a circle on a 2D grid
    :param x: 2D array
    :param y: 2D array
    :param D: diameter 
    :return: a 2D array
    """
    circle = (x**2+y**2)<(D/2)**2
    return circle
