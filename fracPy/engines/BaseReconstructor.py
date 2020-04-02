

class BaseReconstructor(object):
    """
    Common properties for a reconstruction engine can be defined here, for instance a loading and saving method.
    """
    def __init__(self):
        raise NotImplementedError()

    def start_reconstruction(self):
        raise NotImplementedError()
