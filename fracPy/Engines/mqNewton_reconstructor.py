import numpy as np
from matplotlib import pyplot as plt
# fracPy imports
try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Engines.BaseReconstructor import BaseReconstructor
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy.Monitors.Monitor import Monitor
from fracPy.utils.gpuUtils import getArrayModule
from fracPy.utils.utils import fft2c, ifft2c
import logging
import tqdm
import sys

class mqNewton(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Reconstruction_parameters, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('mqNewton')
        self.logger.info('Sucesfully created momentum accelerated qNewton engine')

        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        self.initializeReconstructionParams()
        # initialize momentum
        self.optimizable.initializeObjectMomentum()
        self.optimizable.initializeProbeMomentum()
        # set object and probe buffers
        self.optimizable.objectBuffer = self.optimizable.object.copy()
        self.optimizable.probeBuffer = self.optimizable.probe.copy()
        self.params.momentumAcceleration = True
        
        
    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the qNewton settings.
        :return:
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1
        self.beta1 = 0.5
        self.beta2 = 0.5
        self.betaProbe_m = 0.25
        self.betaObject_m = 0.25
        self.feedbackM = 0.3          # feedback
        self.frictionM = 0.7          # friction
        self.momentum_method = 'ADAM' # which optimizer to use for momentum updates
        
    def initializeAdaptiveMomentum(self):
        self.momentum_engine = getattr(mqNewton, self.momentum_method)
        print("Momentum Engines implemented: momentum, ADAM, NADAM")
        print("Momentum engine used: {}".format(self.momentum_method))
        if self.momentum_method in ['ADAM', 'NADAM']:
            # 2nd order momentum terms
            self.optimizable.objectMomentum_v = self.optimizable.objectMomentum.copy()
            self.optimizable.probeMomentum_v = self.optimizable.probeMomentum.copy()
            
        
    def doReconstruction(self):
        self._prepareReconstruction()
        self.initializeAdaptiveMomentum()
        
        self.pbar = tqdm.trange(self.numIterations, desc='mqNewton', file=sys.stdout, leave=True)
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.optimizable.positions[positionIndex]
                sy = slice(row, row + self.optimizable.Np)
                sx = slice(col, col + self.optimizable.Np)
                # note that object patch has size of probe array
                objectPatch = self.optimizable.object[..., sy, sx].copy()
                
                # make exit surface wave
                self.optimizable.esw = objectPatch * self.optimizable.probe
                
                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                # object update
                self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.optimizable.probe = self.probeUpdate(objectPatch, DELTA)
                
                # momentum updates
                self.objectMomentumUpdate(loop)
                self.probeMomentumUpdate(loop)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0
            #todo clearMemory implementation
    
    def ADAM(self, grad, mt, vt, itr):
        xp = getArrayModule(grad)
        beta1_scale = (1 - self.beta1**itr)
        beta2_scale = (1 - self.beta2**itr)
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        vt = self.beta2 * vt + (1 - self.beta2) * xp.linalg.norm(grad.flatten().squeeze(), 2)**2
        m_hat = mt / beta1_scale
        v_hat = vt / beta2_scale
        return m_hat / (v_hat**0.5 + 1e-8), mt, vt  

    def NADAM(self, grad, mt, vt, itr):
        """
        NADAM optimizer uses adaptive momentum updates (ADAM) with Nesterov 
        momentum acceleration
        :return:
        """
        xp = getArrayModule(grad)

        beta1_scale = (1 - self.beta1**itr)
        beta2_scale = (1 - self.beta2**itr)
    
        norm_sq = xp.linalg.norm(grad.flatten(),2)**2
        mt = self.beta1 * mt + (1 - self.beta1) * grad
        vt = self.beta2 * vt + (1 - self.beta2) * norm_sq
        m_hat = mt / beta1_scale
        v_hat = vt / beta2_scale
        update = (self.beta1 * m_hat + grad*(1 - self.beta1)/beta1_scale) / (v_hat**0.5 + 1e-8)
        return update, mt, vt 
    
    
    def momentum(self, grad, mt, vt, itr):
        """
        standard momentum update
        :return:
        """
        mt = grad + self.frictionM * mt
        return mt, mt, vt
    
    
    def objectMomentumUpdate(self, loop):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.optimizable.objectBuffer - self.optimizable.object
        update, self.optimizable.objectMomentum, self.optimizable.objectMomentum_v =\
            self.momentum_engine(self, gradient, self.optimizable.objectMomentum, self.optimizable.objectMomentum_v, loop+1)
            
        self.optimizable.object -= self.betaObject_m * update
        self.optimizable.objectBuffer = self.optimizable.object.copy()


    def probeMomentumUpdate(self, loop):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.optimizable.probeBuffer - self.optimizable.probe
        update, self.optimizable.probeMomentum, self.optimizable.probeMomentum_v =\
            self.momentum_engine(self, gradient, self.optimizable.probeMomentum, self.optimizable.probeMomentum_v, loop+1)
            
        self.optimizable.probe -= self.betaProbe_m * update
        self.optimizable.probeBuffer = self.optimizable.probe.copy()


    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        xp = getArrayModule(objectPatch)
        Pmax = xp.max(xp.sum(xp.abs(self.optimizable.probe), axis=(0, 1, 2, 3)))
        frac = xp.abs(self.optimizable.probe)/Pmax * self.optimizable.probe.conj() / (xp.abs(self.optimizable.probe)**2 + self.regObject)
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0,2,3), keepdims=True)

       
    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        xp = getArrayModule(objectPatch)
        Omax = xp.max(xp.sum(xp.abs(self.optimizable.object), axis=(0, 1, 2, 3)))
        frac = xp.abs(objectPatch)/Omax * objectPatch.conj() /  (xp.abs(objectPatch)**2 + self.regProbe)
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0,1,3), keepdims=True)
        return r
