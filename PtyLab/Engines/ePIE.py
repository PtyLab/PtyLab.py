import logging
import sys

import numpy as np
import tqdm
from matplotlib import pyplot as plt

from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.Params.Params import Params

# PtyLab imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.utils.gpuUtils import getArrayModule
from PtyLab.utils.utils import fft2c, ifft2c


class ePIE(BaseEngine):
    r"""
    Extended Ptychographical Iterative Engine (ePIE).

    ePIE jointly reconstructs the complex object and illumination probe by
    iterating over overlapping scan positions.[^maiden2009] For each position $j$, the
    exit surface wave is formed as

    $$
    \Psi_j = O_j P
    $$

    where $O_j$ is the object patch illuminated by the probe $P$.

    After propagation to the detector plane and application of the measured
    intensity constraint, the corrected exit wave $\Psi'_j$ is propagated
    back to the object plane. The resulting exit-wave difference is

    $$
    \Delta\Psi_j = \Psi'_j - \Psi_j
    $$

    In the classical single-mode ePIE formulation, the object and probe are
    updated according to

    $$
    O'_j = O_j + \beta_O \frac{P^*}{\max |P|^2}\Delta\Psi_j
    $$

    and

    $$
    P' = P + \beta_P \frac{O_j^*}{\max |O_j|^2}\Delta\Psi_j
    $$

    where $\beta_O$ and $\beta_P$ control the object and probe update step
    sizes.

    The PtyLab implementation generalizes these updates to its multidimensional
    reconstruction representation by summing the corresponding contributions
    over the relevant wavelength, mode, and slice dimensions.

    The default ePIE settings are `betaObject = 0.25`,
    `betaProbe = 0.25`, and `numIterations = 50`.

    [^maiden2009]: A. M. Maiden and J. M. Rodenburg,
        "An improved ptychographical phase retrieval algorithm for diffractive
        imaging," Ultramicroscopy 109, 1256-1262 (2009).
        https://doi.org/10.1016/j.ultramic.2009.05.012
    """
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("ePIE")
        self.logger.info("Sucesfully created ePIE ePIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Initialize ePIE-specific reconstruction parameters.

        Defaults:
            betaObject (float):
                Object update step size. Default is 0.25.
            betaProbe (float):
                Probe update step size. Default is 0.25.
            numIterations (int):
                Number of reconstruction iterations. Default is 50.
        """
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50

    def reconstruct(self, experimentalData: ExperimentalData = None):
        """
        Run the ePIE reconstruction to completion.

        This method consumes the generator returned by `reconstruct_stepwise()`
        until all reconstruction iterations and scan positions have been
        processed.

        Use `reconstruct_stepwise()` when custom operations need to be inserted
        between individual scan-position updates.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use for the reconstruction. If provided,
                it replaces the currently attached experimental data.
        """
        for _ in self.reconstruct_stepwise(experimentalData):
            pass

    def reconstruct_stepwise(self, experimentalData: ExperimentalData = None):
        r"""
        Run the ePIE reconstruction one scan-position update at a time.

        For each reconstruction iteration, the scan positions are visited in the
        order selected by `params.positionOrder`. At each position $j$, the
        corresponding object patch is extracted and combined with the current
        probe to form the exit surface wave:

        $$
        \Psi_j = O_j P
        $$

        The exit wave is propagated to the detector plane, constrained by the
        measured diffraction intensity through `intensityProjection()`, and
        propagated back to obtain an updated exit wave $\Psi'_j$.

        The exit-wave correction is

        $$
        \Delta\Psi_j = \Psi'_j - \Psi_j
        $$

        By default, the object is updated using the standard ePIE rule implemented
        by `objectPatchUpdate()`.

        If `params.objectTVregSwitch` is enabled, the TV-regularized update
        `objectPatchUpdate_TV()` is used every `params.objectTVfreq` iterations.
        The standard ePIE object update is retained and an additional TV
        regularization term is added with strength controlled by
        `params.objectTVregStepSize`.

        The probe is updated using `probeUpdate()` after each object update.

        If `params.OPRP` is enabled, position-dependent probe estimates are
        retrieved from `reconstruction.probe_storage` before each scan-position
        update and stored again after the probe update. Without OPRP, a shared
        probe estimate is updated sequentially across all scan positions.

        After all scan positions in an iteration have been processed,
        `getErrorMetrics()` evaluates the reconstruction error and
        `applyConstraints()` applies the enabled reconstruction constraints.

        The method yields after every scan-position update, allowing custom code
        to be interleaved with the reconstruction.

        Args:
            experimentalData (ExperimentalData, optional):
                Experimental dataset to use for the reconstruction. If provided,
                it replaces the currently attached experimental data.

        Yields:
            tuple:
                `(iteration, positionLoop)` after each scan-position update.
        """
        if experimentalData is not None:
            self.reconstruction.data = experimentalData
            self.experimentalData = experimentalData
        self._prepareReconstruction()

        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(
            self.numIterations, desc="ePIE", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            if self.params.OPRP:
                # make the initial guess the default storage
                self.reconstruction.probe_storage.push(
                    self.reconstruction.probe,
                    0,
                    self.experimentalData.ptychogram.shape[0],
                )
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                if self.params.OPRP:
                    self.reconstruction.probe = self.reconstruction.probe_storage.get(
                        positionIndex
                    )
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
                if (
                    self.params.objectTVregSwitch
                    and loop % self.params.objectTVfreq == 0
                ):
                    object_patch = self.objectPatchUpdate_TV(objectPatch, DELTA)
                else:
                    object_patch = self.objectPatchUpdate(objectPatch, DELTA)

                self.reconstruction.object[..., sy, sx] = object_patch
                
                #self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                #    objectPatch, DELTA
                #)

                # probe update
                self.reconstruction.probe = self.probeUpdate(objectPatch, DELTA)
                if self.params.OPRP:
                    self.reconstruction.probe_storage.push(
                        self.reconstruction.probe,
                        positionIndex,
                        self.experimentalData.ptychogram.shape[0],
                    )
                yield loop, positionLoop

            # get error metric
            self.getErrorMetrics()

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            # self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the object patch using the ePIE object-update rule.

        For the classical single-mode case, the probe weighting is

        $$
        W_P(x,y) = \frac{P^*(x,y)}{\max_{x,y}|P(x,y)|^2}
        $$

        and the object patch is updated according to

        $$
        O'_j = O_j + \beta_O W_P\Delta\Psi_j
        $$

        where $O_j$ is the current object patch, $\Delta\Psi_j$ is the
        exit-wave correction obtained from the detector-plane intensity
        constraint, and $\beta_O$ is `betaObject`.

        In the multidimensional PtyLab representation, contributions from the
        relevant wavelength, probe-mode, and slice dimensions are summed before
        updating the object patch.

        Args:
            objectPatch (ndarray):
                Current object patch at the active scan position.
            DELTA (ndarray):
                Exit-wave correction
                `reconstruction.eswUpdate - reconstruction.esw`.

        Returns:
            ndarray:
                Updated object patch.
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)

        frac = self.reconstruction.probe.conj() / xp.max(
            xp.sum(xp.abs(self.reconstruction.probe) ** 2, axis=(0, 1, 2, 3))
        )
        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=(0, 2, 3), keepdims=True
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        r"""
        Update the probe using the ePIE probe-update rule.

        For the classical single-mode case, the object weighting is

        $$
        W_O(x,y) = \frac{O_j^*(x,y)}{\max_{x,y}|O_j(x,y)|^2}
        $$

        and the probe is updated according to

        $$
        P' = P + \beta_P W_O\Delta\Psi_j
        $$

        where $O_j$ is the current object patch, $\Delta\Psi_j$ is the
        exit-wave correction obtained from the detector-plane intensity
        constraint, and $\beta_P$ is `betaProbe`.

        In the multidimensional PtyLab representation, the implemented update
        sums the corresponding correction over axes `(0, 1, 3)` while preserving
        the probe-mode dimension.

        Args:
            objectPatch (ndarray):
                Current object patch at the active scan position.
            DELTA (ndarray):
                Exit-wave correction
                `reconstruction.eswUpdate - reconstruction.esw`.

        Returns:
            ndarray:
                Updated probe.
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / xp.max(
            xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3))
        )
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r
