import logging

class Params(object):
    """
    Some settings are shared in between different optimizers, such as the type of propagatorType that you intend to use,
    if you want to use probe orthogonalization, etc. These are stored in the reconstruction_parameters object.

    This ensures that code like this will work as expected:

    optimizer1 = optimizers.
    """
    def __init__(self):
        # datalogger


        self.logger = logging.getLogger('Params')

        # Default settings for switches, settings that involve how things are computed
        self.fftshiftSwitch = False
        # this is an internal setting, tracking wether of not the fftshifts have been done. Do not change this yourself
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

        self.l2reg = False # l2 regularisation
        self.l2reg_probe_aleph = 0.01 # strength of the regularizer
        self.l2reg_object_aleph = 0.001

        # autofocusing
        self.TV_autofocus = True
        self.TV_autofocus_intensityonly = False
        self.TV_autofocus_stepsize = 5
        self.TV_autofocus_aleph = 0.01
        self.TV_autofocus_roi = [0.4, 0.6]
        self.TV_autofocus_range_dof = 11
        self.TV_autofocus_friction = 0.7
        # only run every run_every iterations
        self.TV_autofocus_run_every = 3

        # map a change in positions to a change in z. Experimental, do not use
        self.map_position_to_z_change = True

        self.OPRP = True


