import numpy as np
from matplotlib import pyplot as plt
import tqdm
from typing import Any
from scipy.interpolate import interp2d
from fracPy.utils.visualisation import hsvplot

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
from fracPy.Params.Params import Params
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor
from fracPy.operators.operators import aspw
import logging
import sys


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

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the ePIE settings.
        :return:
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.aPIEfriction = 0.7
        self.feedback = 0.5

        if not hasattr(self.optimizable, 'thetaMomentum'):
            self.optimizable.thetaMomentum = 0
        if not hasattr(self.optimizable, 'thetaHistory'):
            self.optimizable.thetaHistory = np.array([])

        self.thetaSearchRadiusMin = 0.01
        self.thetaSearchRadiusMax = 0.1
        self.ptychogramUntransformed = self.experimentalData.ptychogram.copy()
        self.experimentalData.W = np.ones_like(self.optimizable.Xd)

        if self.optimizable.theta == None:
            raise ValueError('theta value is not given')

    def doReconstruction(self):
        self._prepareReconstruction()

        xp = getArrayModule(self.optimizable.object)

        # linear search
        thetaSearchRadiusList = np.linspace(self.thetaSearchRadiusMax, self.thetaSearchRadiusMin,
                                            self.numIterations)

        self.pbar = tqdm.trange(self.numIterations, desc='aPIE', file=sys.stdout, leave=True)
        for loop in self.pbar:
            # save theta search history
            self.optimizable.thetaHistory = np.append(self.optimizable.thetaHistory,
                                                      asNumpyArray(self.optimizable.theta))

            # select two angles (todo check if three angles behave better)
            theta = np.array([self.optimizable.theta, self.optimizable.theta + thetaSearchRadiusList[loop] *
                              (-1 + 2 * np.random.rand())]) + self.optimizable.thetaMomentum

            # save object and probe
            probeTemp = self.optimizable.probe.copy()
            objectTemp = self.optimizable.object.copy()

            # probe and object buffer (todo maybe there's more elegant way )
            probeBuffer = xp.zeros_like(probeTemp)  # shape=(np.array([probeTemp, probeTemp])).shape)
            probeBuffer = [probeBuffer, probeBuffer]
            objectBuffer = xp.zeros_like(
                objectTemp)  # , shape=(np.array([objectTemp, objectTemp])).shape)  # for polychromatic case this will need to be multimode
            objectBuffer = [objectBuffer, objectBuffer]
            # initialize error
            errorTemp = np.zeros((2, 1))

            for k in range(2):
                self.optimizable.probe = probeTemp
                self.optimizable.object = objectTemp
                # reset ptychogram (transform into estimate coordinates)
                Xq = T_inv(self.optimizable.Xd, self.optimizable.Yd, self.optimizable.zo,
                           theta[k])  # todo check if 1D is enough to save time
                for l in range(self.experimentalData.numFrames):
                    temp = self.ptychogramUntransformed[l]
                    f = interp2d(self.optimizable.xd, self.optimizable.xd, temp, kind='linear', fill_value=0)
                    temp2 = abs(f(Xq[0], self.optimizable.xd))
                    temp2 = np.nan_to_num(temp2)
                    temp2[temp2 < 0] = 0
                    self.optimizable.ptychogram[l] = xp.array(temp2)

                # renormalization(for energy conservation) # todo not layer by layer?
                self.experimentalData.ptychogram = self.experimentalData.ptychogram / np.linalg.norm(
                    self.experimentalData.ptychogram) * np.linalg.norm(self.ptychogramUntransformed)

                self.experimentalData.W = np.ones_like(self.optimizable.Xd)
                fw = interp2d(self.optimizable.xd, self.optimizable.xd, self.experimentalData.W, kind='linear',
                              fill_value=0)
                self.experimentalData.W = abs(fw(Xq[0], self.optimizable.xd))
                self.experimentalData.W = np.nan_to_num(self.experimentalData.W)
                self.experimentalData.W[self.experimentalData.W == 0] = 1e-3
                self.experimentalData.W = xp.array(self.optimizable.W)

                # todo check if it is right
                if self.params.fftshiftSwitch:
                    self.experimentalData.ptychogram = xp.fft.ifftshift(self.experimentalData.ptychogram, axes=(-1, -2))
                    self.experimentalData.W = xp.fft.ifftshift(self.experimentalData.W, axes=(-1, -2))

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
                line = plt.plot(0, self.optimizable.theta, 'o-')[0]
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

    def xtotiltU(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray, wavelength: np.ndarray):
        """
        go from detector plane coordinates to corresponding  tilted plane spatial frequency coordinates
        :param x: detector x coordinates
        :param y: detector y coordinates
        :param theta: tilt angle between sample plane and detector plane in degrees
        :params wavelength: illumination wavelength
        :return: warped spatial frequencies u,v associated with x,y
        """
        theta = toRadians(theta)

        ro = np.sqrt(x ** 2 + y ** 2 + self.experimentalData.zo ** 2)
        v = y
        u = (x * np.cos(theta) - np.sin(theta) * (ro - self.experimentalData.zo)) / (wavelength * ro)

        return u, v

    def tiltUtoX(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, wavelength: np.ndarray):
        """
        go from tilted plane spatial frequency coordinates to corresponding detector plane coordinates
        :param u: spatial frequency coordinates associated with x in sample coordinates that are tilted with respect to
        detector coordinates
        :param v: spatial frequency coordinates associated with y in sample coordinates that are tilted with respect to detector
        coordinates
        :param theta: tilt angle between sample plane and detector plane in degrees
        :param wavelength:illumination wavelength
        :return:
        :rtype: x,y(the detector coordinates associated with the input spatial frequencies, after transform inversion)
        """
        # for derivation see ~placeholder for pdf
        theta = toRadians(theta)
        a = 1. + ((u / v) + np.sin(theta) / (v * wavelength) / np.cos(theta)) ** 2 - (v * wavelength) ** (-2)
        b = -(2. * np.sin(theta) * self.experimentalData.zo / (v * np.cos(theta)) ** 2) * (u + np.sin(theta)) / \
            wavelength
        c = (1 + np.tan(theta))
        y = (-b - np.sign(v) * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        # y is not invertible with this quadratic equation at v=0 as 0/0 is undefined, but y=lambda*r0*v, so y=0
        y = np.where(v == 0, y, 0)
        # at a=0 we again gets something that is not defined, furthermore, the numerical accuracy due to rounding
        # error s becomes problematic when a=very small, so we replace by the linear equation for those cases.
        y = np.where(a < 1.e-6, y, -c / b)
        # for x its less trivial to choose the right sign for the solution,
        # calculate the spatial frequencies where x=0, this marks a ux0(y) line. all coordinates that have a u(x,
        # y) that is larger are positive, all that are smaller have a negative x
        #

        rx0 = np.sqrt(self.experimentalData.zo ** 2 + y ** 2)
        ux0 = np.sin(theta) * (self.experimentalData.zo - rx0) / (wavelength * rx0)
        x = np.sign(u - ux0) * np.real(np.sqrt(((v * wavelength) ** (-2) - 1) * y ** 2 - self.zo ** 2))
        # when y=0,v=0 this equation is not defined so we invert for y=0, and get another quadratic equation for x
        ax = (np.cos(2 * theta) - 2 * wavelength * u * np.sin(theta) - (wavelength * u) ** 2)
        bx = 2 * self.experimentalData.zo * np.sin(theta) * np.cos(theta)
        self.experimentalData.zo = self.experimentalData.zo
        cx = -(self.experimentalData.zo ** 2 * ((wavelength * u) * (2 * np.sin(theta) + (wavelength * u))))
        x2 = -bx + np.sqrt(bx ** 2 - 4 * ax * cx) / (2 * ax)
        x2 = np.where(np.abs(ax) < 1.e-6, -cx / bx, x2)
        x = np.where(np.abs(y) < 1.e-6, x2, y)
        x = np.where(np.abs(u - ux0) < 1.e-6, 0, x)
        return x, y
    def xtoU(self,x,y,wavelength):
        r0 = np.sqrt(x ** 2 + y ** 2 + self.experimentalData.zo ** 2)
        

def toRadians(theta):
    theta =theta*np.pi/180.
    return theta