import logging

import numpy as np


def set_gpuSwitch():
    try:
        import cupy

        if cupy.cuda.is_available():
            print(
                "cupy and cuda available, switching to GPU for faster reconstruction."
            )
            return True
        else:
            print(
                "cuda unavailable or incompatible, switching to CPU for reconstruction."
            )
            return False
    except:
        print("cupy unavailable, switching to CPU for reconstruction.")
        return False


class Params(object):
    """
    Some settings are shared in between different optimizers, such as the type of propagatorType that you intend to use,
    if you want to use probe orthogonalization, etc. These are stored in the reconstruction_parameters object.

    This ensures that code like this will work as expected:

    optimizer1 = optimizers.
    """

    def __init__(self):
        # datalogger
        # Total variation regularizerion options
        self.positionCorrectionSwitch_radius = 1
        self.objectTVfreq = 5
        self.objectTVregSwitch = False
        self.objectTVregStepSize = 1e-3

        # other stuff
        self.weigh_probe_updates_by_intensity = False
        self.logger = logging.getLogger("Params")

        # Default settings for switches, settings that involve how things are computed
        self.fftshiftSwitch = False
        # this is an internal setting, tracking wether of not the fftshifts have been done. Do not change this yourself
        self.fftshiftFlag = False
        self.FourierMaskSwitch = False
        self.CPSCswitch = False
        self.CPSCupsamplingFactor = None

        self.intensityConstraint = "standard"  # standard or sigmoid
        self.propagatorType = "Fraunhofer"  # 'Fresnel' 'ASP'
        self.momentumAcceleration = False  # default False, it is turned on in the individual Engines that use momentum
        self.adaptiveMomentumAcceleration = False  # default False, it is turned on in the individual Engines that use momentum

        ## Specific reconstruction settings that are the same for all Engines
        self.gpuSwitch = set_gpuSwitch()
        # This only makes sense on a GPU, not there yet
        self.saveMemory = False
        self.probeUpdateStart = 1
        self.objectUpdateStart = 1
        self.positionOrder = "random"  # 'random' or 'sequential' or 'NA'

        ## Swtiches used in applyConstraints method:
        self.orthogonalizationSwitch = False
        self.orthogonalizationFrequency = 10  # probe orthogonalization frequency
        # object regularization
        self.objectSmoothenessSwitch = False
        self.objectSmoothenessWidth = (
            2  # # pixels over which object is assumed fairly smooth
        )
        self.objectSmoothnessAleph = (
            1e-2  # relaxation constant that determines strength of regularization
        )
        self.absObjectSwitch = False  # force the object to be abs-only
        self.absObjectBeta = 1e-2  # relaxation parameter for abs-only constraint
        self.objectContrastSwitch = False  # pushes object to zero outside ROI
        # probe regularization
        self.probeSmoothenessSwitch = False  # enforce probe smootheness
        self.probeSmoothnessAleph = 5e-2  # relaxation parameter for probe smootheness
        self.probeSmoothenessWidth = 3  # loose object support diameter
        self.probeBoundary = False  # probe cut-off based on a window
        self.absorbingProbeBoundary = (
            False  # controls if probe has period boundary conditions (zero)
        )
        self.absorbingProbeBoundaryAleph = 5e-2
        self.probePowerCorrectionSwitch = False  # probe normalization to measured PSD
        self.modulusEnforcedProbeSwitch = False  # enforce empty beam

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
        self.adaptiveDenoisingSwitch = (
            False  # estimated noise floor to be clipped from raw data
        )

        self.l2reg = False  # l2 regularisation
        self.l2reg_probe_aleph = 0.01  # strength of the regularizer
        self.l2reg_object_aleph = 0.001

        # autofocusing
        # Wether or not to perform TV autofocusing
        self.TV_autofocus = False
        # what to focus: can be 'TV', 'std', 'min_std', or a callable
        self.TV_autofocus_metric = "TV"
        # Only look at the TV of the intensity as a focusing metric
        self.TV_autofocus_intensityonly = False
        # stepsize
        self.TV_autofocus_stepsize = 5
        # ???
        self.TV_autofocus_aleph = 0.01
        # Region of interest. Can either be in pixels or in a fraction of No/ Np
        self.TV_autofocus_roi = [0.4, 0.6]
        # Propagation range in depths of focus
        self.TV_autofocus_range_dof = 11
        # Friction ot the step algorithm
        self.TV_autofocus_friction = 0.7
        # Shat to focus, can be either 'object' or 'probe'
        self.TV_autofocus_what = "object"
        # only run every run_every iterations
        self.TV_autofocus_run_every = 3
        # minimum distance, set to None for no limit
        self.TV_autofocus_min_z = None
        # maximum distance, set to None for no limit
        self.TV_autofocus_max_z = None
        # number of planes to examine
        self.TV_autofocus_nplanes = 11

        # map a change in positions to a change in z. Experimental, do not use
        self.map_position_to_z_change = False

        # Default values of all OPR parameters
        # Index of the incoherent probe modes that are linked in a subspace.
        self.OPR_modes = np.array([0])
        # Size of the subspace which is used for the truncated SVD
        self.OPR_subspace = 4
        # Feedback parameter of the OPR modes. The higher the more the probes are allowed
        # to evolve freely. Value range: [0, 1]
        self.OPR_alpha = 0.05
        # Every x iterations the tv constraint is used
        # CURRENTLY not implementd
        self.OPR_tv_freq = 5
        # feedback parameter for the tv constraint
        # CURRENTLY not implementd
        self.OPR_tv = False
        # truncated SVD to chose, either standard numpy svd or randomized tsvd, which
        # saves some computational time
        self.OPR_tsvd_type = "numpy"  # numpy or randomized
        # Switch to orthogonolize all incoherent probe modes
        self.OPR_orthogonalize_modes = True
        # If set True only slowly changing probes are allowed
        self.OPR_neighbor_constraint = False

        # SHG stuff
        self.SHG_probe = False
