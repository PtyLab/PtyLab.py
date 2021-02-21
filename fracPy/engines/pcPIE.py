import numpy as np
from matplotlib import pyplot as plt
from fracPy.utils import gpuUtils

try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.utils.utils import fft2c, ifft2c
import logging


class pcPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, monitor)
        self.logger = logging.getLogger('pcPIE')
        self.logger.info('Successfully created pcPIE pcPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        # initialize pcPIE params
        self.initializeReconstructionParams()
        # initialize momentum
        self.optimizable.initializeObjectMomentum()
        self.optimizable.initializeProbeMomentum()
        # set object and probe buffers
        self.optimizable.objectBuffer = self.optimizable.object.copy()
        self.optimizable.probeBuffer = self.optimizable.probe.copy()

        self.momentumAcceleration = True
        
    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the pcPIE settings.
        :return:
        """
        # these are same as mPIE
        # self.eswUpdate = self.optimizable.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1     # probe regularization
        self.alphaObject = 0.1    # object regularization
        self.betaM = 0.3          # feedback
        self.stepM = 0.7          # friction
        self.probeWindow = np.abs(self.optimizable.probe)

    def doReconstruction(self):
        self._prepareReconstruction()

        # actual reconstruction ePIE_engine
        import tqdm
        for loop in tqdm.tqdm(range(self.numIterations)):
            # set position order
            self.setPositionOrder()

            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.experimentalData.positions[positionIndex]
                sy = slice(row, row + self.experimentalData.Np)
                sx = slice(col, col + self.experimentalData.Np)
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
                if self.positionCorrectionSwitch:
                    self.positionCorrection(objectPatch, positionIndex, sy, sx)
                # position correction
                # xp = getArrayModule(objectPatch)
                # if len(self.optimizable.error) > self.startAtIteration:
                #     # position gradients
                #     # shiftedImages = xp.zeros((self.rowShifts.shape + objectPatch.shape))
                #     cc = xp.zeros((len(self.rowShifts), 1))
                #     for shifts in range(len(self.rowShifts)):
                #         tempShift = xp.roll(objectPatch, self.rowShifts[shifts], axis=-2)
                #         # shiftedImages[shifts, ...] = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                #         shiftedImages = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                #         cc[shifts] = xp.squeeze(xp.sum(shiftedImages.conj() * self.optimizable.object[..., sy, sx],
                #                                        axis=(-2, -1)))
                #     # truncated cross - correlation
                #     # cc = xp.squeeze(xp.sum(shiftedImages.conj() * self.optimizable.object[..., sy, sx], axis=(-2, -1)))
                #     cc = abs(cc)
                #     betaGrad = 1000
                #     normFactor = xp.sum(objectPatch.conj() * objectPatch, axis=(-2, -1)).real
                #     grad_x = betaGrad * xp.sum((cc.T - xp.mean(cc)) / normFactor * xp.array(self.colShifts))
                #     grad_y = betaGrad * xp.sum((cc.T - xp.mean(cc)) / normFactor * xp.array(self.rowShifts))
                #     r = 3
                #     if abs(grad_x) > r:
                #         grad_x = r * grad_x / abs(grad_x)
                #     if abs(grad_y) > r:
                #         grad_y = r * grad_y / abs(grad_y)
                #     self.D[positionIndex, :] = self.daleth * gpuUtils.asNumpyArray([grad_y, grad_x]) + self.beth *\
                #                                self.D[positionIndex, :]


                # momentum updates todo: make this every T iteration?
                # Todo @lars explain this
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()
            # if len(self.optimizable.error) > self.startAtIteration:
            #     # update positions
            #     self.experimentalData.encoder = (self.experimentalData.positions - self.adaptStep * self.D -
            #                                      self.experimentalData.No//2 + self.experimentalData.Np//2) * \
            #                                                 self.experimentalData.dxo
            #     # fix center of mass of positions
            #     self.experimentalData.encoder[:, 0] = self.experimentalData.encoder[:, 0] - \
            #                                         np.mean(self.experimentalData.encoder[:, 0]) + self.meanEncoder00
            #     self.experimentalData.encoder[:, 1] = self.experimentalData.encoder[:, 1] - \
            #                                         np.mean(self.experimentalData.encoder[:, 1]) + self.meanEncoder01
            #
            #     # self.experimentalData.positions[:,0] = self.experimentalData.positions[:,0] - \
            #     #         np.round(np.mean(self.experimentalData.positions[:,0]) -
            #     #                   np.mean(self.experimentalData.positions0[:,0]) )
            #     # self.experimentalData.positions[:, 1] = self.experimentalData.positions[:, 1] - \
            #     #                                         np.around(np.mean(self.experimentalData.positions[:, 1]) -
            #     #                                                   np.mean(self.experimentalData.positions0[:, 1]))
            #
            #     # show reconstruction
            #     if (len(self.optimizable.error) > self.startAtIteration) & (np.mod(loop,
            #                                                 self.monitor.figureUpdateFrequency) == 0):
            #         figure, ax = plt.subplots(1, 1, num=102, squeeze=True, clear=True, figsize=(5, 5))
            #         ax.set_title('Estimated scan grid positions')
            #         ax.set_xlabel('(um)')
            #         ax.set_ylabel('(um)')
            #         # ax.set_xscale('symlog')
            #         plt.plot(self.experimentalData.positions0[:, 1] * self.experimentalData.dxo * 1e6,
            #                  self.experimentalData.positions0[:, 0] * self.experimentalData.dxo * 1e6, 'bo')
            #         plt.plot(self.experimentalData.positions[:, 1] * self.experimentalData.dxo * 1e6,
            #                  self.experimentalData.positions[:, 0] * self.experimentalData.dxo * 1e6, 'yo')
            #         # plt.xlabel('(um))')
            #         # plt.ylabel('(um))')
            #         # plt.show()
            #         plt.tight_layout()
            #         plt.show(block=False)
            #
            #         figure2, ax2 = plt.subplots(1, 1, num=103, squeeze=True, clear=True, figsize=(5, 5))
            #         ax2.set_title('Displacement')
            #         ax2.set_xlabel('(um)')
            #         ax2.set_ylabel('(um)')
            #         plt.plot(self.D[:, 1] * self.experimentalData.dxo * 1e6,
            #                  self.D[:, 0] * self.experimentalData.dxo * 1e6, 'o')
            #         # ax.set_xscale('symlog')
            #         plt.tight_layout()
            #         plt.show(block=False)
            #
            #     # elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
            #         figure.canvas.draw()
            #         figure.canvas.flush_events()
            #         figure2.canvas.draw()
            #         figure2.canvas.flush_events()
            #         self.showReconstruction(loop)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

            #todo clearMemory implementation

    def objectMomentumUpdate(self):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.optimizable.objectBuffer - self.optimizable.object
        self.optimizable.objectMomentum = gradient + self.stepM * self.optimizable.objectMomentum
        self.optimizable.object = self.optimizable.object - self.betaM * self.optimizable.objectMomentum
        self.optimizable.objectBuffer = self.optimizable.object.copy()


    def probeMomentumUpdate(self):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.optimizable.probeBuffer - self.optimizable.probe
        self.optimizable.probeMomentum = gradient + self.stepM * self.optimizable.probeMomentum
        self.optimizable.probe = self.optimizable.probe - self.betaM * self.optimizable.probeMomentum
        self.optimizable.probeBuffer = self.optimizable.probe.copy()


    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absP2 = xp.abs(self.optimizable.probe)**2
        Pmax = xp.max(xp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        if self.experimentalData.operationMode =='FPM':
            frac = abs(self.optimizable.probe)/Pmax*\
                   self.optimizable.probe.conj()/(self.alphaObject*Pmax+(1-self.alphaObject)*absP2)
        else:
            frac = self.optimizable.probe.conj()/(self.alphaObject*Pmax+(1-self.alphaObject)*absP2)
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=2, keepdims=True)

       
    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absO2 = xp.abs(objectPatch) ** 2
        Omax = xp.max(xp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        frac = objectPatch.conj() / (self.alphaProbe * Omax + (1-self.alphaProbe) * absO2)
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=1, keepdims=True)
        return r


