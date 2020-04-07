
class Params:
    """An empty class to have the same form of obj.params as in MATLAB.

    See ptylab.initialParams for all the switches and properties.

    """

    def __init__(self):
        # these parameters should be implemented in the reconstructor object.
        self.numIterations = 1 # number of iterations
        self.figureUpdateFrequency = 1
        self.probeWindow = None # no idea
        self.obj = None

        return