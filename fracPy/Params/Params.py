import logging

class Params(object):
    """
    Some settings are shared in between different optimizers, such as the type of propagatorType that you intend to use,
    if you want to use probe orthogonalization, etc. These are stored in the reconstruction_parameters object.

    This ensures that code like this will work as expected:

    optimizer1 = optimizers.
    """
    # To prevent adding attributes in an uncontrolled way, we freeze the attributes after init.
    _isFrozen = False

    def __init__(self):
        # datalogger
        self.logger = logging.getLogger('Params')

        # Default settings for switches, settings that involve how things are computed
        self.fftshiftSwitch = False
        self.fftshiftFlag = False
        self.FourierMaskSwitch = False
        self.CPSCswitch = False
        self.CPSCupsamplingFactor = None

        self.intensityConstraint = 'standard'  # standard or sigmoid
        self.propagatorType = 'Fraunhofer'  # 'Fresnel' 'ASP'
        self.momentumAcceleration = False  # default False, it is turned on in the individual Engines that use momentum
        self.adaptiveMomentumAcceleration = False  # default False, it is turned on in the individual Engines that use momentum

        ## Specific reconstruction settings that are the same for all Engines
        self.gpuSwitch = False
        self.gpuFlag = 0
        # This only makes sense on a GPU, not there yet
        self.saveMemory = False
        self.probeUpdateStart = 1
        self.objectUpdateStart = 1
        self.positionOrder = 'random'  # 'random' or 'sequential' or 'NA'

        ## Swtiches used in applyConstraints method:
        self.orthogonalizationSwitch = False
        self.orthogonalizationFrequency = 10  # probe orthogonalization frequency
        # object regularization
        self.objectSmoothenessSwitch = False
        self.objectSmoothenessWidth = 2  # # pixels over which object is assumed fairly smooth
        self.objectSmoothnessAleph = 1e-2  # relaxation constant that determines strength of regularization
        self.absObjectSwitch = False  # force the object to be abs-only
        self.absObjectBeta = 1e-2  # relaxation parameter for abs-only constraint
        self.objectContrastSwitch = False  # pushes object to zero outside ROI
        # probe regularization
        self.probeSmoothenessSwitch = False  # enforce probe smootheness
        self.probeSmoothnessAleph = 5e-2  # relaxation parameter for probe smootheness
        self.probeSmoothenessWidth = 3  # loose object support diameter
        self.probeBoundary = False  # probe cut-off based on a window
        self.absorbingProbeBoundary = False  # controls if probe has period boundary conditions (zero)
        self.absorbingProbeBoundaryAleph = 5e-2
        self.probePowerCorrectionSwitch = False  # probe normalization to measured PSD
        self.modulusEnforcedProbeSwitch = False  # enforce empty beam
        self.comStabilizationSwitch = False  # center of mass stabilization for probe
        self.absProbeSwitch = False  # force the probe to be abs-only
        self.absProbeBeta = 1e-2  # relaxation parameter for abs-only constraint
        # other
        self.couplingSwitch = False  # couple adjacent wavelengths
        self.couplingAleph = 50e-2  # couple adjacent wavelengths (relaxation parameter)
        self.binaryProbeSwitch = False  # enforce probe to be binary
        self.binaryProbeThreshold = 0.1  # binarize threshold
        self.binaryProbeAleph = 10e-2  # relaxation parameter for binary constraint
        self.backgroundModeSwitch = False  # background estimate
        self.comStabilizationSwitch = False  # center of mass stabilization for probe
        self.PSDestimationSwitch = False
        self.objectContrastSwitch = False  # pushes object to zero outside ROI
        self.positionCorrectionSwitch = False  # position correction for encoder
        self.adaptiveDenoisingSwitch = False  # estimated noise floor to be clipped from raw data

        # To prevent adding attributes in an uncontrolled way
        self._isFrozen = True

    # rewrite the __setattr__ method to prevent adding attributes in an uncontrolled way
    def __setattr__(self, key, value):
        if self._isFrozen and not hasattr(self, key):
            raise TypeError('Params does not have this attribute, check spelling!')
        object.__setattr__(self, key, value)


