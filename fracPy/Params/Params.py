import numpy as np
import warnings
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.monitors.Monitor import Monitor
from fracPy.utils.utils import ifft2c, fft2c, orthogonalizeModes, circ
from fracPy.operators.operators import aspw, scaledASP
import cupy as cp
import logging

class Params(object):
    def __init__(self):#, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # These statements don't copy any data, they just keep a reference to the object
        # self.optimizable = optimizable
        # self.experimentalData = experimentalData
        # self.monitor = monitor
        # self.monitor.optimizable = optimizable

        # datalogger
        self.logger = logging.getLogger('Params')
        
        # Default settings for switches, settings that involve how things are computed
        self.fftshiftSwitch = False
        self.fftshiftFlag = False
        self.FourierMaskSwitch = False
        self.CPSCswitch = False
        self.fontSize = 17
        self.intensityConstraint = 'standard'  # standard or sigmoid
        self.propagator = 'Fraunhofer'  # 'Fresnel' 'ASP'
        self.momentumAcceleration = False  # default False, it is turned on in the individual engines that use momentum


        ## Specific reconstruction settings that are the same for all engines
        self.gpuSwitch = False
        # This only makes sense on a GPU, not there yet
        self.saveMemory = False
        self.probeUpdateStart = 1
        self.objectUpdateStart = 1
        self.positionOrder = 'random'  # 'random' or 'sequential'

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
        self.probeSmoothenessSwitch = False # enforce probe smootheness
        self.probeSmoothnessAleph = 5e-2  # relaxation parameter for probe smootheness
        self.probeSmoothenessWidth = 3  # loose object support diameter
        self.probeBoundary = False # probe cut-off based on a window
        self.absorbingProbeBoundary = False  # controls if probe has period boundary conditions (zero)
        self.absorbingProbeBoundaryAleph = 5e-2
        self.probePowerCorrectionSwitch = False  # probe normalization to measured PSD
        self.modulusEnforcedProbeSwitch = False  # enforce empty beam
        self.comStabilizationSwitch = False  # center of mass stabilization for probe
        self.absProbeSwitch = False  # force the probe to be abs-only
        self.absProbeBeta = 1e-2   # relaxation parameter for abs-only constraint
        # other
        self.couplingSwitch = False  # couple adjacent wavelengths
        self.couplingAleph = 50e-2  # couple adjacent wavelengths (relaxation parameter)
        self.binaryWFSSwitch = False  # enforce WFS to be positive
        self.binaryWFSAleph = 10e-2  # relaxation parameter for binary constraint
        self.backgroundModeSwitch = False  # background estimate
        self.comStabilizationSwitch = True  # center of mass stabilization for probe
        self.PSDestimationSwitch = False
        self.objectContrastSwitch = False  # pushes object to zero outside ROI
        self.positionCorrectionSwitch = False  # position correction for encoder



