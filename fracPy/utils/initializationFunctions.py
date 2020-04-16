import numpy as np

def initialProbeOrObject(shape, type_of_init):
    """
    Implements line 148 of initialParams.m
    :param shape:
    :param type_of_init:
    :return:
    """
    if type(type_of_init) is np.ndarray: # it has already been implemented
        return type_of_init
    if type_of_init not in ['circ', 'rand', 'gaussian', 'ones']:
        raise NotImplementedError()
    if type_of_init == 'ones':
        # NB this is how it's implemented, there's a bit of noise
        shape = np.asarray(shape)
        #print(np.array(shape))
        return np.ones(shape) + 0.001 * np.random.rand(*shape)


