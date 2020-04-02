from pathlib import Path
from utils import *
from reconstruct import reconstruct

dataFolder = Path(r'D:\ptyLab\ptyLabExport')
fileName = 'recent.pkl'

obj = load(dataFolder.joinpath(fileName))

## manual params

obj.params.objectPlot = 'complex'  # 'complex', 'piComlex', 'abs', 'angle', or 'piAngle'
obj.params.intensityConstraint = 'standard'  # 'standard', 'exponential', fluctuation

# engine
obj.params.engine = 'ePIE'  # 'ePIE', 'mPIE', 'zPIE, 'e3PIE', 'm3PIE', 'pcPIE', 'kPIE', 'OPRP', 'SD', 'msPIE', 'sDR'

# main parameters
obj.params.figureUpdateFrequency = 1  # frequency of reconstruction monitor
obj.params.numIterations = 1000  # total number of iterations
obj.params.betaObject = 0.25  # gradient step size object
obj.params.betaProbe = 0.25  # gradient step size probe
obj.params.npsm = 1  # number of orthogonal modes
obj.params.FourierMaskSwitch = False  # apply mask to corrupted pixels
obj.params.gpuSwitch = True  # gpuSwitch

# object regularization 
obj.params.objectSmoothenessSwitch = False  # if  True, impose smootheness
obj.params.objectSmoothenessWidth = 2  # # pixels over which object is assumed fairly smooth
obj.params.objectSmoothnessAleph = 1e-2  # relaxation constant that determines strength of regularization
obj.params.absObjectSwitch = False  # force the object to be abs-only
obj.params.absObjectBeta = 1e-2  # relaxation parameter for abs-only constraint
obj.params.objectContrastSwitch = False  # pushes object to zero outside ROI

# probe regularization 
obj.params.probeSmoothenessSwitch = False  # enforce probe smootheness
obj.params.probeSmoothnessAleph = 5e-2  # relaxation parameter for probe smootheness
obj.params.probeSmoothenessWidth = 3  # loose object support diameter
obj.params.absorbingProbeBoundary = False  # controls if probe has period boundary conditions (zero)
obj.params.probePowerCorrectionSwitch = True  # probe normalization to measured PSD
obj.params.modulusEnforcedProbeSwitch = False  # enforce empty beam
obj.params.comStabilizationSwitch = True  # center of mass stabilization for probe

# other parameters
obj.params.positionOrder = 'random'  # 'sequential' or 'random'
obj.propagator.type = 'Fraunhofer'  # specify propagator between sample and detector (Fraunhofer, Fresnel, ASP, scaledASP)
obj.params.backgroundModeSwitch = False  # background estimate
obj.params.makeGIF = False  # export GIF animation of reconstruction
obj.params.orthogonalizationFrequency = 10  # # iterations until orthogonalization
obj.params.noprpsm = 4  # number of modes for for (i)OPRP
obj.params.verboseLevel = 'high'  # control how much output is plotted
obj.params.fftshiftSwitch = True
# obj.params.probeROI = [1 obj.Np 1 obj.Np]
obj.params.objectZoom = 1
obj.monitor.objectMax = 1

obj = reconstruct(obj)

## export data
