class reconstruction(object):
    """
    This is the core of fracPy. It contains the object with all possible features. Any new settings should go into this object.

    Settings for the reconstruction are all specified in the params object

    Settings for Monitors (called Monitor) are all specified in monitor

    Settings for
    """

    def __init__(self, **kwargs):
        self.load_default_settings()
        self.update_settings(**kwargs)

    def load_default_settings(self):
        self.params = ReconstructionParams()
        self.monitor = Monitor()


class Monitor(object):
    """
    Settings that involve the data Monitors
    """

    def __init__(self):
        self.objectPlot = 'complex'
        # TODO implement all options here as well


class ReconstructionParams(object):
    """
    This class contains the settings for a particular reconstruction
    """

    def __init__(self):
        # frequency of the reconstruction monitor
        self.figure_update_frequency = 20
        # total number of iterations
        self.num_iterations = 1000
        # gradient step size object
        self.beta_object = 0.25
        # TODO implement all of the options
        self.gpu_switch = True
