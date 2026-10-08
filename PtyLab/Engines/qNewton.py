import numpy as np
from matplotlib import pyplot as plt

# PtyLab imports
try:
    import cupy as cp
except ImportError:
    # print("Cupy not available, will not be able to run GPU based computation")
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None
import logging
import sys

import tqdm

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import getArrayModule
from PtyLab.utils.utils import fft2c, ifft2c


class qNewton(BaseEngine):
    r"""
    Experimental quasi-Newton-inspired ptychographic reconstruction engine.

    `qNewton` provides an alternative to the standard ePIE and mPIE update
    rules by using locally weighted, regularized object and probe corrections.
    Its purpose is to adapt the update strength to the local illumination or
    object amplitude instead of applying the same normalization everywhere.

    For the object update, the current implementation uses a factor of the form

    $$
    \frac{|P|}{P_{\max}}\frac{P^*}{|P|^2+\lambda_O}
    $$

    while the probe update uses the corresponding object-dependent expression,

    $$
    \frac{|O|}{O_{\max}}\frac{O^*}{|O|^2+\lambda_P}
    $$

    where `regObject` and `regProbe` provide the regularization terms.

    The inverse-intensity factors reduce the update in strongly illuminated
    regions and prevent excessively large corrections where the local probe or
    object intensity becomes small. The additional amplitude weighting further
    modulates the contribution of weakly illuminated or weak-object regions.

    This type of locally scaled update may be useful __when the illumination or
    object amplitude varies strongly across the reconstruction__ and a standard
    globally normalized PIE update becomes poorly balanced. 

    The remaining reconstruction workflow follows the standard PIE structure:
    the exit surface wave is formed from the object and probe,
    `intensityProjection()` applies the measured diffraction constraint, and
    the resulting exit-wave correction is used to update the object and probe.

    Notes:
        This engine is retained as an experimental algorithm implementation.
        Its normalization and theoretical formulation should be validated
        against the original method or reference before further development.
    """
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("qNewton")
        self.logger.info("Sucesfully created qNewton engine")

        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Initialize parameters specific to the experimental qNewton reconstruction.

        This method sets the default object and probe update strengths,
        regularization parameters, and number of reconstruction iterations.
        
        The object and probe updates are controlled by `betaObject` and
        `betaProbe`, while `regObject` and `regProbe` regularize the local
        inverse-intensity factors used in the qNewton-inspired update rules.
        """
        self.betaProbe = 1
        self.betaObject = 1
        self.regObject = 1
        self.regProbe = 1
        self.numIterations = 50

    def reconstruct(self, experimentalData: ExperimentalData = None):
        """
        Run the qNewton ptychographic reconstruction.

        The reconstruction follows the standard PIE workflow: each scan position
        is processed sequentially, `intensityProjection()` applies the measured
        diffraction constraint, and the resulting exit-wave correction is used to
        update the object and probe.

        The qNewton-specific behavior is provided by `objectPatchUpdate()` and
        `probeUpdate()`, which apply the locally weighted and regularized update
        rules defined by this engine.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use for the reconstruction. If provided,
                it replaces the dataset currently attached to the engine.
        """
        if experimentalData is not None:
            self.reconstruction.data = experimentalData
            self.experimentalData = experimentalData
        self._prepareReconstruction()

        self.pbar = tqdm.trange(
            self.numIterations, desc="qNewton", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()

            for positionLoop, positionIndex in enumerate(self.positionIndices):
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
                self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                    objectPatch, DELTA
                )

                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

            # todo clearMemory implementation

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the object patch using the qNewton-inspired correction.

        The update is weighted by the current probe according to

        $$
        \frac{|P|}{P_{\max}}\frac{P^*}{|P|^2+\lambda_O}
        $$

        where `regObject` provides the regularization term $\lambda_O$.

        Args:
            objectPatch (np.ndarray):
                Current object patch at the scan position.

            DELTA (np.ndarray):
                Exit-wave correction after the intensity projection.

        Returns:
            np.ndarray:
                Updated object patch.
        """
       
        xp = getArrayModule(objectPatch)
        Pmax = xp.max(xp.sum(xp.abs(self.reconstruction.probe), axis=(0, 1, 2, 3)))
        frac = (
            xp.abs(self.reconstruction.probe)
            / Pmax
            * self.reconstruction.probe.conj()
            / (xp.abs(self.reconstruction.probe) ** 2 + self.regObject)
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2, 3), keepdims=True
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the probe using the qNewton-inspired correction.

        The update is weighted by the current object patch according to

        $$
        \frac{|O|}{O_{\max}}\frac{O^*}{|O|^2+\lambda_P}
        $$

        where `regProbe` provides the regularization term $\lambda_P$.

        Args:
            objectPatch (np.ndarray):
                Current object patch at the scan position.

            DELTA (np.ndarray):
                Exit-wave correction after the intensity projection.

        Returns:
            np.ndarray:
                Updated probe.
        """
        xp = getArrayModule(objectPatch)
        Omax = xp.max(xp.sum(xp.abs(self.reconstruction.object), axis=(0, 1, 2, 3)))
        frac = (
            xp.abs(objectPatch)
            / Omax
            * objectPatch.conj()
            / (xp.abs(objectPatch) ** 2 + self.regProbe)
        )
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r
