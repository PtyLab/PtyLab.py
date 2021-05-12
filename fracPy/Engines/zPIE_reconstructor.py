import numpy as np
from matplotlib import pyplot as plt
import tqdm
from fracPy.utils.visualisation import hsvplot

try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

# fracPy imports
from fracPy.Optimizables.Optimizable import Optimizable
from fracPy.Engines.BaseReconstructor import BaseReconstructor
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData
from fracPy.Params.ReconstructionParameters import Reconstruction_parameters
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.Monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging
import sys


class zPIE(BaseReconstructor):

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Reconstruction_parameters, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('zPIE')
        self.logger.info('Sucesfully created zPIE zPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.DoF = self.optimizable.DoF
        self.zPIEgradientStepSize = 100  #gradient step size for axial position correction (typical range [1, 100])
        self.zPIEfriction = 0.7
        self.focusObject = True
        self.zMomentun = 0


    def doReconstruction(self):
        self._prepareReconstruction()

        ###################################### actual reconstruction zPIE_engine #######################################

        xp = getArrayModule(self.optimizable.object)
        if not hasattr(self.optimizable, 'zHistory'):
            self.optimizable.zHistory = []

        # preallocate grids
        if self.params.propagator == 'ASP':
            n = self.optimizable.Np.copy()
        else:
            n = 2*self.optimizable.Np

        if not self.focusObject:
            n = self.optimizable.Np

        X,Y = xp.meshgrid(xp.arange(-n//2, n//2), xp.arange(-n//2, n//2))
        w = xp.exp(-(xp.sqrt(X**2+Y**2)/self.optimizable.Np)**4)

        self.pbar = tqdm.trange(self.numIterations, desc='zPIE', file=sys.stdout, leave=True)  # in order to change description to the tqdm progress bar
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            # get positions
            if loop == 1:
                zNew = self.optimizable.zo.copy()
            else:
                d = 10
                dz = np.linspace(-d * self.DoF, d * self.DoF, 11)/10
                merit = []
                # todo, mixed states implementation, check if more need to be put on GPU to speed up
                for k in np.arange(len(dz)):
                    if self.focusObject:
                        imProp, _ = aspw(w * xp.squeeze(self.optimizable.object[...,
                                                        (self.optimizable.No // 2 - n // 2):(self.optimizable.No // 2 + n // 2),
                                                        (self.optimizable.No // 2 - n // 2):(self.optimizable.No // 2 + n // 2)]),
                                         dz[k], self.optimizable.wavelength, n * self.optimizable.dxo)
                    else:
                        if self.optimizable.nlambda==1:
                            imProp, _ = aspw(xp.squeeze(self.optimizable.probe[..., :, :]),
                                             dz[k], self.optimizable.wavelength, self.optimizable.Lp)
                        else:
                            nlambda = self.optimizable.nlambda//2
                            imProp, _ = aspw(xp.squeeze(self.optimizable.probe[nlambda,...,:,:]),
                                            dz[k], self.optimizable.spectralDensity[nlambda], self.optimizable.Lp)

                    # TV approach
                    aleph = 1e-2
                    gradx = imProp-xp.roll(imProp, 1, axis=1)
                    grady = imProp-xp.roll(imProp, 1, axis=0)
                    merit.append(asNumpyArray(xp.sum(xp.sqrt(abs(gradx)**2+abs(grady)**2+aleph))))

                feedback = np.sum(dz*merit)/np.sum(merit)    # at optimal z, feedback term becomes 0
                self.zMomentun = self.zPIEfriction*self.zMomentun+self.zPIEgradientStepSize*feedback
                zNew = self.optimizable.zo+self.zMomentun

            self.optimizable.zHistory.append(self.optimizable.zo)

            # print updated z
            self.pbar.set_description('zPIE: update z = %.3f mm (dz = %.1f um)' % (self.optimizable.zo*1e3, self.zMomentun*1e6))

            # reset coordinates
            self.optimizable.zo = zNew


            # re-sample is automatically done by using @property
            if self.params.propagator != 'ASP':
                self.optimizable.dxp = self.optimizable.wavelength*self.optimizable.zo\
                                            /self.optimizable.Ld
                # reset propagator
                self.optimizable.quadraticPhase = xp.array(np.exp(1.j * np.pi/(self.optimizable.wavelength * self.optimizable.zo)
                                                                  * (self.optimizable.Xp**2 + self.optimizable.Yp**2)))
        ##################################################################################################################


            for positionLoop, positionIndex in enumerate(self.positionIndices):
                ### patch1 ###
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

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            if loop == 0:
                figure, ax = plt.subplots(1, 1, num=666, squeeze=True, clear=True, figsize=(5, 5))
                ax.set_title('Estimated distance (object-camera)')
                ax.set_xlabel('iteration')
                ax.set_ylabel('estimated z (mm)')
                ax.set_xscale('symlog')
                line = plt.plot(0, zNew, 'o-')[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(0, np.log10(len(self.optimizable.zHistory)-1), np.minimum(len(self.optimizable.zHistory), 100))
                idx = np.rint(10**idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.optimizable.zHistory)[idx]*1e3)
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(np.min(self.optimizable.zHistory)*1e3, np.max(self.optimizable.zHistory)*1e3)

                figure.canvas.draw()
                figure.canvas.flush_events()
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.optimizable.probe.conj() / xp.max(xp.sum(xp.abs(self.optimizable.probe) ** 2, axis=(0, 1, 2, 3)))
        return objectPatch + self.betaObject * xp.sum(frac * DELTA, axis=(0, 2, 3), keepdims=True)

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3)))
        r = self.optimizable.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r


