import numpy as np
from matplotlib import pyplot as plt
import tqdm
from typing import Any
from scipy.interpolate import interp2d
from scipy import ndimage
from fracPy.utils.visualisation import hsvplot
from fracPy.utils.test.CoordinateTransformations import xtoU, tiltUtoX, xtoTiltU, quickshow
from cupyx.scipy import ndimage as cuNdimage
# fracPy imports
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines.BaseReconstructor import BaseReconstructor
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Params.Params import Params
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging
import sys
import time


try:
    import cupy as cp
except ImportError:
    print('Cupy not available, will not be able to run GPU based computation')
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None


class aPIE(BaseReconstructor):
    """
    aPIE: angle correction PIE: ePIE combined with Luus-Jaakola algorithm (the latter for angle correction) + momentum
    """

    def __init__(self, optimizable: Optimizable, experimentalData: ExperimentalData, params: Params, monitor: Monitor):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to aPIE reconstruction
        super().__init__(optimizable, experimentalData, params, monitor)
        self.logger = logging.getLogger('aPIE')
        self.logger.info('Sucesfully created aPIE aPIE_engine')
        self.logger.info('Wavelength attribute: %s', self.optimizable.wavelength)
        self.initializeReconstructionParams()
        self.prepareUVgrid()

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.params.aPIEflag = True
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.aPIEfriction = 0.7
        self.feedback = 0.5

        if not hasattr(self.optimizable, 'thetaMomentum'):
            self.optimizable.thetaMomentum = 0
        if not hasattr(self.optimizable, 'thetaHistory'):
            self.optimizable.thetaHistory = np.array([])
        self.warp_axis = 0  # which axis needs tilt correction(z-warp_axis defines the sample reflection plane)
        self.thetaSearchRadiusMin = 0.01
        self.thetaSearchRadiusMax = 0.1
        self.ptychogramUntransformed = self.experimentalData.ptychogram.copy()
        self.experimentalData.W = np.ones_like(self.optimizable.Xd)
        self.ePIEloops = 3  # number of loops to perform with epie with both test angles before choosing which one is
        # better, making this higher makes it slower, but more stable, I would advise to set it atleast at two,
        # interpolation bottlenecks the speed anyway otherwise.
        if self.optimizable.theta == None:
            raise ValueError('theta value is not given')
        self.Xd=self.optimizable.Xd.copy()
        self.Yd=self.optimizable.Yd.copy()
        self.dxd=self.optimizable.dxd.copy()
        self.zo=self.optimizable.zo
    def doReconstruction(self):
        self._prepareReconstruction()

        xp = getArrayModule(self.optimizable.object)

        # linear search
        thetaSearchRadiusList = np.linspace(self.thetaSearchRadiusMax, self.thetaSearchRadiusMin,
                                            self.numIterations)
        xp2 = getArrayModule(self.Uq)
        self.pbar = tqdm.trange(self.numIterations, desc='aPIE', file=sys.stdout, leave=True)
        for loop in self.pbar:
            # save theta search history
            self.optimizable.thetaHistory = np.append(self.optimizable.thetaHistory,
                                                      asNumpyArray(self.optimizable.theta))

            # select two angles (todo check if three angles behave better)
            theta = xp2.array([self.optimizable.theta, self.optimizable.theta + thetaSearchRadiusList[loop] *
                               (-1 + 2 * np.random.rand())]) + self.optimizable.thetaMomentum

            # save object and probe
            probeTemp = self.optimizable.probe.copy()
            objectTemp = self.optimizable.object.copy()

            # probe and object buffer (todo maybe there's more elegant way )
            probeBuffer = xp.zeros_like(probeTemp)  # shape=(np.array([probeTemp, probeTemp])).shape)
            probeBuffer = [probeBuffer, probeBuffer]
            objectBuffer = xp.zeros_like(
                objectTemp)  # , shape=(np.array([objectTemp, objectTemp])).shape)  # for polychromatic case this
            # will need to be multimode
            objectBuffer = [objectBuffer, objectBuffer]
            # initialize error
            errorTemp = np.zeros((2, 1))

            for k in range(2):
                self.optimizable.probe = probeTemp
                self.optimizable.object = objectTemp
                # reset ptychogram (transform into estimate coordinates)
                Xqcalc, Yqcalc = tiltUtoX(self.Uq, self.Vq, self.optimizable.zo, self.wavelength,
                                  theta[k], axis=self.warp_axis, output_list=1)
                Xq = ((Xqcalc - xp2.amin(self.Xd)) / self.dxd).astype(np.float32)
                Yq = ((Yqcalc - xp2.amin(self.Yd)) / self.dxd).astype(np.float32)

                for l in range(self.experimentalData.numFrames):
                    temp = self.ptychogramUntransformed[l]
                    # f = interp2d(self.optimizable.xd, self.optimizable.xd, temp, kind='linear', fill_value=0)
                    # temp2 = abs(f(Xq, Yq))
                    if self.params.gpuSwitch is True:
                        temp2 = xp2.reshape(cuNdimage.map_coordinates(temp, xp2.array([Yq,
                                                                                                 Xq]), order=0),
                                            (self.optimizable.Nd, self.optimizable.Nd))
                    else:
                        temp2 = xp2.reshape(ndimage.map_coordinates(temp, xp2.array([Yq,
                                                                                     Xq]), order=0),
                                            (self.optimizable.Nd, self.optimizable.Nd))
                    temp2 = xp2.nan_to_num(temp2)
                    temp2[temp2 < 0] = 0
                    self.experimentalData.ptychogram[l] = xp.array(temp2)

                # renormalization(for energy conservation) # todo not layer by layer?
                self.experimentalData.ptychogram = self.experimentalData.ptychogram / xp2.linalg.norm(
                    self.experimentalData.ptychogram) * xp2.linalg.norm(self.ptychogramUntransformed)

                self.experimentalData.W = xp2.ones_like(self.Xd)
                if self.params.gpuSwitch is True:
                    self.experimentalData.W = xp2.nan_to_num(
                        xp2.reshape(cuNdimage.map_coordinates(self.experimentalData.W, xp2.array([Yq,
                                                                                                Xq]), order=0),
                                    (self.optimizable.Nd,
                                     self.optimizable.Nd)))
                else:
                    self.experimentalData.W = xp2.nan_to_num(
                        xp2.reshape(cuNdimage.map_coordinates(self.experimentalData.W, xp2.array([Yq,
                                                                                                  Xq]), order=0),
                                    (self.optimizable.Nd,
                                     self.optimizable.Nd)))

                self.experimentalData.W[self.experimentalData.W == 0] = 1e-3

                # todo check if it is right
                if self.params.fftshiftSwitch:
                    self.experimentalData.ptychogram = xp.fft.ifftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                    self.experimentalData.W = xp.fft.ifftshift(self.experimentalData.W, axes=(-1, -2))
                for n in range(self.ePIEloops):
                    # set position order
                    self.setPositionOrder()

                    for positionLoop, positionIndex in enumerate(self.positionIndices):
                        ### patch1 ###
                        # get object patch1
                        row1, col1 = self.optimizable.positions[positionIndex]
                        sy = slice(row1, row1 + self.optimizable.Np)
                        sx = slice(col1, col1 + self.optimizable.Np)
                        # note that object patch has size of probe array
                        objectPatch = self.optimizable.object[..., sy, sx].copy()

                        # make exit surface wave
                        self.optimizable.esw = objectPatch * self.optimizable.probe

                        # propagate to camera, intensityProjection, propagate back to object
                        self.intensityProjection(positionIndex)

                        # difference term1
                        DELTA = self.optimizable.eswUpdate - self.optimizable.esw

                        # object update
                        self.optimizable.object[..., sy, sx] = self.objectPatchUpdate(objectPatch, DELTA)

                        # probe update
                        self.optimizable.probe = self.probeUpdate(objectPatch, DELTA)

                    # get error metric
                    self.getErrorMetrics()
                    # remove error from error history
                    errorTemp[k] = self.optimizable.error[-1]
                    self.optimizable.error = np.delete(self.optimizable.error, -1)

                    # apply Constraints
                    self.applyConstraints(loop)
                # update buffer
                probeBuffer[k] = self.optimizable.probe
                objectBuffer[k] = self.optimizable.object

            if errorTemp[1] < errorTemp[0]:
                dtheta = theta[1] - theta[0]
                self.optimizable.theta = theta[1]
                self.optimizable.probe = probeBuffer[1]
                self.optimizable.object = objectBuffer[1]
                self.optimizable.error = np.append(self.optimizable.error, errorTemp[1])
            else:
                dtheta = 0
                self.optimizable.theta = theta[0]
                self.optimizable.probe = probeBuffer[0]
                self.optimizable.object = objectBuffer[0]
                self.optimizable.error = np.append(self.optimizable.error, errorTemp[0])

            self.optimizable.thetaMomentum = self.feedback * dtheta + self.aPIEfriction * self.optimizable.thetaMomentum
            # print updated theta
            self.pbar.set_description('aPIE: update a=%.3f deg (search radius=%.3f deg, thetaMomentum=%.3f deg)'
                                      % (self.optimizable.theta, thetaSearchRadiusList[loop],
                                         self.optimizable.thetaMomentum))

            # show reconstruction
            if loop == 0:
                figure, ax = plt.subplots(1, 1, num=777, squeeze=True, clear=True, figsize=(5, 5))
                ax.set_title('Estimated angle')
                ax.set_xlabel('iteration')
                ax.set_ylabel('estimated theta [deg]')
                ax.set_xscale('symlog')
                line = plt.plot(0, asNumpyArray(self.optimizable.theta), 'o-')[0]
                plt.tight_layout()
                plt.show(block=False)

            elif np.mod(loop, self.monitor.figureUpdateFrequency) == 0:
                idx = np.linspace(0, np.log10(len(self.optimizable.thetaHistory) - 1),
                                  np.minimum(len(self.optimizable.thetaHistory), 100))
                idx = np.rint(10 ** idx).astype('int')

                line.set_xdata(idx)
                line.set_ydata(np.array(self.optimizable.thetaHistory)[idx])
                ax.set_xlim(0, np.max(idx))
                ax.set_ylim(min(self.optimizable.thetaHistory), max(self.optimizable.thetaHistory))

                figure.canvas.draw()
                figure.canvas.flush_events()

            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info('switch to cpu')
            self._move_data_to_cpu()
            self.params.gpuFlag = 0
        self.params.aPIEflag=False
        # self.thetaSearchRadiusMax = thetaSearchRadiusList[loop]

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

    def prepareUVgrid(self, interP_samplingDensity=None):
        """
        Generate a Spatial frequency space grid used in the reconstruction, find the largest spatial frequencies
        associated with the detector grid and then create an equally sampled grid that includes those spatial
        frequencies, and sets the sample space pixel sizes according to that spatial frequency range.

        """
        if not hasattr(self, 'Ugrid'):
            if interP_samplingDensity is None:
                interP_samplingDensity = self.optimizable.Nd
            Fdetx, Fdety = xtoU(self.optimizable.Xd, self.optimizable.Yd, self.optimizable.zo,
                                self.experimentalData.wavelength)
            lowerbound = np.amin(Fdetx)
            upperbound = np.amax(np.amax(Fdetx))
            self.Ugrid = np.linspace(lowerbound, upperbound, interP_samplingDensity, dtype=np.float64)

            self.Uq, self.Vq = np.meshgrid(self.Ugrid, self.Ugrid, sparse=False)
            print(self.optimizable.dxo)
            self.optimizable.dxp = 1 / abs(upperbound - lowerbound)
            print(self.optimizable.dxo)
            self.latest_know_z = self.optimizable.zo
        else:
            if self.latest_know_z != self.optimizable.zo:
                x = getArrayModule(self.Uq)

                if interP_samplingDensity is None:
                    interP_samplingDensity = self.optimizable.Nd
                Fdetx, Fdety = xtoU(self.optimizable.Xd, self.optimizable.Yd, self.optimizable.zo,
                                    self.experimentalData.wavelength)
                lowerbound = x.amin(Fdetx)
                upperbound = x.amax(x.amax(Fdetx))
                self.Ugrid = x.linspace(lowerbound, upperbound, interP_samplingDensity, dtype=np.float64)

                self.Uq, self.Vq = x.meshgrid(self.Ugrid, self.Ugrid, sparse=False)
                self.optimizable.dxp = 1 / abs(upperbound - lowerbound)
                self.latest_know_z = self.optimizable.zo
