parser = {}

parser['absorbingProbeBoundary'] = False  # set probe to zero outside 2 x exit pupil diameter
parser['betaObject'] = 0.25  # mPIE feedback parameter (object)
parser['betaProbe'] = 0.25  # mPIE feedback parameter (probe)
parser['comStabilizationSwitch'] = True  # if True, the probe is centered
parser['deleteFrames'] = []  # 1D vector with numbers indicating which frames to exclude
parser['ePIE_engine'] = 'ePIE'  # plot every ... iterations
parser['fftshiftSwitch'] = True  # switch to prevent excessive fftshifts
parser['figureUpdateFrequency'] = 1  # plot every ... iterations
parser['FourierMaskSwitch'] = False  # switch to apply Fourier mask
parser['initialObject'] = 'ones'  # choose initial object ('ones', 'rand')
parser['initialProbe'] = 'circ'  # choose initial probe ('obes', 'rand')
parser['intensityConstraint'] = 'standard'  # ('standard', 'sigmoid')
parser['npsm'] = 1  # number of probe state mixtures
parser['nosm'] = 1  # number of object state mixtures
parser['numIterations'] = 1  # number of iterations
parser['objectPlot'] = 'complex'  # 'complex', 'abs', 'angle', piAngle
parser['objectUpdateStart'] = 1  # iteration when object update starts (makes algorithm stable if good initial guess is known)
parser['positionOrder'] = 'random'  # position order for sequential solvers ('sequential' or 'random')
parser['probePowerCorrectionSwitch'] = True  # switch to indicate if probe power correction is applied
parser['saveMemory'] = True  # if True, then the algorithm is run with low memory consumption (e.g. detector error is not saved)
parser['singleSwitch'] = True  # if True, data is converted to single to save memory and processing time
parser['smoothenessSwitch'] = False  # if True, apply smoothness constraint to object
parser['gpuSwitch'] = False  # if True, then algorithm runs on gpu
parser['orthogonalizationFrequency'] = 10  # probe orthogonalization frequency
parser['batchSize'] = 10  # fft-block size for parallel engines
parser['makeGIF'] = False  # probe orthogonalization frequency
parser['fontSize'] = 17

# todo: declare default params for e3PIE
