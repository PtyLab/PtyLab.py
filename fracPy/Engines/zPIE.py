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
from fracPy.Reconstruction.Reconstruction import Reconstruction
from fracPy.Engines.BaseEngine import BaseEngine
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Params.Params import Params
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.Monitor.Monitor import Monitor
from fracPy.Operators.Operators import aspw
import logging
import sys


class zPIE(BaseEngine):

    def __init__(self, reconstruction: Reconstruction, experimentalData: ExperimentalData, params: Params, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger('zPIE')
        self.logger.info('Sucesfully created zPIE zPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.DoF = self.reconstruction.DoF
        self.zPIEgradientStepSize = 100  #gradient step size for axial position correction (typical range [1, 100])
        self.zPIEfriction = 0.7
        self.focusObject = True
        self.zMomentun = 0

    def show_defocus(self, viewer=None, scanrange_times_dof=1000, N_points=10):
        z = np.linspace(-1,1,N_points) * scanrange_times_dof * self.reconstruction.DoF

        from fracPy.Operators.Operators import aspw
        reconstruction = self.reconstruction
        defocii = np.abs(np.array(
            [aspw(reconstruction.object, dz, reconstruction.wavelength, reconstruction.Lo)[0] for dz
             in z])**2)


        if viewer is None:
            import napari
            viewer = napari.Viewer()
        viewer.add_image(defocii)


    def reconstruct(self, experimentalData=None, reconstruction=None):
        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)
        self._prepareReconstruction()

        ###################################### actual reconstruction zPIE_engine #######################################

        xp = getArrayModule(self.reconstruction.object)
        if not hasattr(self.reconstruction, 'zHistory'):
            self.reconstruction.zHistory = []

        # preallocate grids
        if self.params.propagatorType == 'ASP':
            n = self.reconstruction.Np*1
        else:
            n = 2*self.reconstruction.Np

        if not self.focusObject:
            n = self.reconstruction.Np

        X,Y = xp.meshgrid(xp.arange(-n//2, n//2), xp.arange(-n//2, n//2))
        w = xp.exp(-(xp.sqrt(X**2+Y**2) / self.reconstruction.Np) ** 4)

        self.pbar = tqdm.trange(self.numIterations, desc='zPIE', file=sys.stdout, leave=True)  # in order to change description to the tqdm progress bar
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            # get positions
            if loop == 1:
                zNew = self.reconstruction.zo.copy()
            else:
                d = 50

                dz = np.linspace(-d * self.DoF, d * self.DoF, 11)/10

                merit = []
                # todo, mixed states implementation, check if more need to be put on GPU to speed up
                for k in np.arange(len(dz)):
                    if self.focusObject:
                        imProp, _ = aspw(w * xp.squeeze(self.reconstruction.object[...,
                                                        (self.reconstruction.No // 2 - n // 2):(self.reconstruction.No // 2 + n // 2),
                                                        (self.reconstruction.No // 2 - n // 2):(self.reconstruction.No // 2 + n // 2)]),
                                         dz[k], self.reconstruction.wavelength, n * self.reconstruction.dxo)
                    else:
                        if self.reconstruction.nlambda==1:
                            imProp, _ = aspw(xp.squeeze(self.reconstruction.probe[..., :, :]),
                                             dz[k], self.reconstruction.wavelength, self.reconstruction.Lp)
                        else:
                            nlambda = self.reconstruction.nlambda // 2
                            imProp, _ = aspw(xp.squeeze(self.reconstruction.probe[nlambda, ..., :, :]),
                                             dz[k], self.reconstruction.spectralDensity[nlambda], self.reconstruction.Lp)

                    # TV approach
                    aleph = 1e-2
                    gradx = imProp-xp.roll(imProp, 1, axis=1)
                    grady = imProp-xp.roll(imProp, 1, axis=0)
                    merit.append(asNumpyArray(xp.sum(xp.sqrt(abs(gradx)**2+abs(grady)**2+aleph))))

                feedback = np.sum(dz*merit)/np.sum(merit)    # at optimal z, feedback term becomes 0
                print('Step size: ', feedback)
                self.zMomentun = self.zPIEfriction*self.zMomentun+self.zPIEgradientStepSize*feedback
                zNew = self.reconstruction.zo + self.zMomentun

            self.reconstruction.zHistory.append(self.reconstruction.zo)

            # print updated z
            self.pbar.set_description('zPIE: update z = %.3f mm (dz = %.1f um)' % (self.reconstruction.zo * 1e3, self.zMomentun * 1e6))

            # reset coordinates
            self.reconstruction.zo = zNew


            # re-sample is automatically done by using @property
            if self.params.propagatorType != 'ASP':
                self.reconstruction.dxp = self.reconstruction.wavelength * self.reconstruction.zo \
                                          / self.reconstruction.Ld
                # reset propagatorType
                # self.reconstruction.quadraticPhase = xp.array(np.exp(1.j * np.pi / (self.reconstruction.wavelength * self.reconstruction.zo)
                #                                                      * (self.reconstruction.Xp ** 2 + self.reconstruction.Yp ** 2)))
        ##################################################################################################################


            for positionLoop, positionIndex in enumerate(self.positionIndices):
                ### patch1 ###
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                # object update
                self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)

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
                idx = np.linspace(0, np.log10(len(self.reconstruction.zHistory) - 1), np.minimum(len(self.reconstruction.zHistory), 100))
                idx = np.rint(10**idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.reconstruction.zHistory)[idx] * 1e3)
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(np.min(self.reconstruction.zHistory) * 1e3, np.max(self.reconstruction.zHistory) * 1e3)

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

        frac = self.reconstruction.probe.conj() / xp.max(xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3)))
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
        r = self.reconstruction.probe + self.betaProbe * xp.sum(frac * DELTA, axis=(0, 1, 3), keepdims=True)
        return r


