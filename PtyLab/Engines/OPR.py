import numpy as np
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    # print('Cupy not available, will not be able to run GPU based computation')
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
from PtyLab.utils.fsvd import rsvd
from PtyLab.utils.gpuUtils import asNumpyArray, getArrayModule, isGpuArray


class OPR(BaseEngine):
    r"""
    Orthogonal Probe Relaxation (OPR) for position-dependent probe reconstruction.

    `OPR` extends conventional ptychographic reconstruction by allowing the
    illumination probe to vary between scan positions while constraining these
    variations to a low-dimensional orthogonal probe subspace.[^odstrcil2016]

    For scan position $j$, the exit surface wave is formed as

    $$
    \Psi_j(\mathbf{r}) = O_j(\mathbf{r}) P_j(\mathbf{r})
    $$

    where $O_j$ is the object patch illuminated at position $j$ and $P_j$ is
    the position-dependent probe.

    Instead of enforcing a single identical probe for all diffraction frames,
    OPR maintains a probe estimate for each scan position. The set of
    position-dependent probes is stored in `reconstruction.probe_stack`.

    At initialization, the selected probe modes defined by `params.OPR_modes`
    are copied to all scan positions. During reconstruction, the probe
    corresponding to the current diffraction frame is loaded from the probe
    stack, updated by the PIE step, and written back to the same scan position.

    Allowing an independent probe at every scan position introduces many
    additional degrees of freedom. OPR therefore constrains the probe
    variations to a low-dimensional subspace. For one selected probe mode,
    the position-dependent probes are arranged as columns of the matrix

    $$
    A = \begin{bmatrix} | & | & & | \\ P_1 & P_2 & \cdots & P_N \\ | & | & & | \end{bmatrix}
    $$

    where each probe is flattened into a vector and $N$ is the number of scan
    positions.

    A truncated singular-value decomposition is used to approximate the probe
    stack as

    $$
    A \approx U_K S_K V_K^{\dagger}
    $$

    where $K =$ `params.OPR_subspace` is the retained subspace
    dimension. Equivalently, the probe at scan position $j$ can be represented
    as a linear combination of a small number of orthogonal probe basis
    functions,

    $$
    P_j(\mathbf{r}) = \sum_{k=1}^{K} c_{k,j}\Phi_k(\mathbf{r})
    $$

    thereby allowing systematic probe variations while suppressing
    unconstrained frame-to-frame fluctuations.

    The low-rank probe estimate is blended with the current probe stack using
    $\alpha =$ `params.OPR_alpha`,

    $$
    P^{\mathrm{new}} = \alpha P^{\mathrm{old}} + (1-\alpha)P^{\mathrm{low-rank}}
    $$

    The truncated decomposition can be computed using the method selected by
    `params.OPR_tsvd_type`. The current implementation supports standard SVD,
    randomized SVD, and a Gram-matrix-based truncated SVD.

    OPR regularization acts at two complementary levels. Despite of a low-rank
    constraint on the position-dependent probe stack, if `params.OPR_neighbor_constraint` is enabled, the subspace
    coefficients $c_{k,j}$ are additionally averaged over neighboring scan
    positions. This imposes smoothness along the scan sequence and suppresses
    abrupt changes in the contribution of each probe basis mode. The two constraints therefore regularize different aspects of the probe
    variation: the low-rank constraint limits which spatial variations are
    allowed, while the neighbor constraint limits how rapidly their coefficients
    may change between successive scan positions.

    OPR can also be combined with mixed-state ptychography.[^eschen2022]
    Multiple mutually incoherent probe modes are used to describe partial
    coherence within each diffraction frame, while OPR accounts for systematic
    changes of these probe modes between scan positions.

    These two descriptions address different probe degrees of freedom. The
    mixed-state model represents the illumination as an incoherent sum of probe
    modes,

    $$
    I_j(\mathbf{q}) = \sum_m \left| \mathcal{F}\left[P_{m,j}(\mathbf{r}) O_j(\mathbf{r})\right] \right|^2
    $$

    where $m$ indexes the mutually incoherent probe modes. OPR additionally
    allows the individual probe modes $P_{m,j}$ to vary with scan position $j$
    while constraining their position dependence to a low-dimensional
    orthogonal subspace.

    In the current implementation, the probe modes included in OPR are selected
    through `params.OPR_modes`. If `params.OPR_orthogonalize_modes` is enabled,
    `orthogonalizeIncoherentModes()` orthogonalizes the incoherent probe modes
    independently at each scan position before the position-dependent probe stack
    is projected onto the OPR subspace.

    If `params.OPR_tv` is enabled, total-variation object updates are applied
    at the interval specified by `params.OPR_tv_freq`.

    [^odstrcil2016]: M. Odstrčil, P. Baksh, S. A. Boden,
            R. Card, J. E. Chad, J. G. Frey, and W. S. Brocklesby,
            "Ptychographic coherent diffractive imaging with orthogonal probe relaxation,"
            Opt. Express 24, 8360-8369 (2016).
            https://doi.org/10.1364/OE.24.008360
    
    [^eschen2022]: W. Eschen, C. Liu, M. Steinert,
            J. Müller, M. P. Ochmann, J. Limpert, and J. Rothhardt,
            "Material-specific high-resolution table-top extreme ultraviolet microscopy,"
            Light Sci. Appl. 11, 117 (2022).
            https://doi.org/10.1038/s41377-022-00797-6
    
    Implementation Notes:
            The current OPR implementation requires CuPy and GPU execution.
            Several probe-stack operations are implemented directly with CuPy
            rather than through the NumPy/CuPy abstraction used elsewhere in
            PtyLab. CPU-only execution is therefore not currently supported.
    """
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("ePIE")
        self.logger.info("Sucesfully created ePIE ePIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        self.initializeReconstructionParams()

    def initializeReconstructionParams(self):
        """
        Initialize parameters specific to the OPR reconstruction.

        The OPR relaxation strength, selected probe modes, and retained
        position-dependent probe subspace dimension are read from
        `params.OPR_alpha`, `params.OPR_modes`, and `params.OPR_subspace`,
        respectively.

        This method also sets the default object and probe update strengths and
        the number of reconstruction iterations. In addition, it defines the
        working-memory budget used by the batched incoherent-mode
        orthogonalization routine.

        The main OPR parameters initialized here are:

        - `alpha`:
        Blending factor between the current probe stack and its low-rank OPR
        approximation.

        - `OPR_modes`:
        Probe-mode indices for which position-dependent probe relaxation is
        applied.

        - `n_subspace`:
        Number of dominant OPR subspace components retained during the
        truncated singular-value decomposition.
        """
        self.alpha = self.params.OPR_alpha
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.numIterations = 50
        self.OPR_modes = self.params.OPR_modes
        self.n_subspace = self.params.OPR_subspace
        # Working-set budget for the chunked batched orthogonalization. Keeps
        # the transpose scratch bounded independently of the frame count.
        self._orthogonalization_chunk_bytes = 128 * 2**20

    def reconstruct(self):
        r"""
        Run the Orthogonal Probe Relaxation reconstruction.

        For each reconstruction iteration, a position-dependent probe is loaded
        from `reconstruction.probe_stack` for the current scan position. The exit
        surface wave is then formed from the local object patch and the selected
        probe, followed by `intensityProjection()` to apply the measured
        diffraction-intensity constraint.

        The resulting exit-wave correction

        $$
        \Delta\Psi_j = \Psi'_j - \Psi_j
        $$

        is used to update the object and probe through `objectPatchUpdate()` and
        `probeUpdate()`. If `params.OPR_tv` is enabled, the object update is
        periodically replaced by `objectPatchUpdate_TV()` according to
        `params.OPR_tv_freq`.

        After each probe update, the selected probe modes are written back to
        `reconstruction.probe_stack` at the corresponding scan position.

        Once all scan positions have been processed, the position-dependent probe
        stack is regularized. If `params.OPR_orthogonalize_modes` is enabled,
        `orthogonalizeIncoherentModes()` first orthogonalizes the incoherent probe
        modes independently at each scan position.

        The complete probe stack is then projected onto a low-dimensional
        position-dependent subspace using `orthogonalizeProbeStack()`, with the
        retained subspace dimension given by `params.OPR_subspace`.

        Finally, reconstruction errors are evaluated with `getErrorMetrics()`,
        the remaining reconstruction constraints are applied with
        `applyConstraints()`, and `showReconstruction()` updates the monitor.

        If GPU acceleration is enabled, the reconstruction data are returned to
        CPU memory after completion.
        """

        self._prepareReconstruction()

        # OPR parameters
        Nmodes = self.OPR_modes.shape[0]
        Np = self.reconstruction.Np
        Nframes = self.experimentalData.numFrames
        mode_slice = self.OPR_modes
        n_subspace = self.n_subspace

        self.reconstruction.probe_stack = cp.zeros(
            (1, 1, Nmodes, 1, Np, Np, Nframes), dtype=cp.complex64
        )

        for i, mode in enumerate(self.OPR_modes):
            # fill the probe-stack with the inital guess of the probes
            self.reconstruction.probe_stack[0, 0, i, 0, :, :, :] = cp.repeat(
                self.reconstruction.probe[0, 0, mode, 0, :, :, cp.newaxis],
                Nframes,
                axis=2,
            )

        # actual reconstruction ePIE_engine
        self.pbar = tqdm.trange(
            self.numIterations, desc="OPR", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            self.it = loop
            # set position order
            self.setPositionOrder()
            for positionLoop, positionIndex in enumerate(self.positionIndices):
                # get object patch
                row, col = self.reconstruction.positions[positionIndex]
                sy = slice(row, row + self.reconstruction.Np)
                sx = slice(col, col + self.reconstruction.Np)
                # note that object patch has size of probe array
                objectPatch = self.reconstruction.object[..., sy, sx].copy()

                # Get dim reduced probe
                self.reconstruction.probe[:, :, mode_slice, :, :, :] = (
                    self.reconstruction.probe_stack[..., positionIndex]
                )

                # make exit surface wave
                self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw

                if loop % self.params.OPR_tv_freq == 0 and self.params.OPR_tv:
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate_TV(
                        objectPatch, DELTA
                    )
                else:
                    # object update
                    self.reconstruction.object[..., sy, sx] = self.objectPatchUpdate(
                        objectPatch, DELTA
                    )

                # probe update
                self.reconstruction.probe = self.probeUpdate(
                    objectPatch, DELTA, weight=1
                )

                # save first, dominant probe mode
                self.reconstruction.probe_stack[..., positionIndex] = cp.copy(
                    self.reconstruction.probe[:, :, mode_slice, :, :, :]
                )

            # get error metric
            self.getErrorMetrics()

            if self.params.OPR_orthogonalize_modes:
                self.orthogonalizeIncoherentModes()

            self.reconstruction.probe_stack = self.orthogonalizeProbeStack(
                self.reconstruction.probe_stack, n_subspace
            )

            # apply Constraints
            self.applyConstraints(loop)

            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

    def orthogonalizeIncoherentModes(self):
        r"""
        Orthogonalize the incoherent probe modes at each scan position.

        For every position in `reconstruction.probe_stack`, the selected probe
        modes are reshaped into a two-dimensional matrix with one row per
        incoherent mode and one column per probe pixel.

        An SVD is then performed,

        $$
        P = U S V^\dagger
        $$

        and the probe modes are replaced by

        $$
        S V^\dagger
        $$

        which provides an equivalent orthogonal representation of the same
        mixed-state probe subspace.

        The orthogonalization is applied independently at every scan position and
        does not alter the position-dependent OPR subspace itself.

        If `params.OPR_fast_orthogonalization` is enabled, the batched
        Gram-matrix implementation `_orthogonalizeIncoherentModes_batched()` is
        used instead of the frame-by-frame SVD. For a probe-mode matrix $P$, the left singular vectors of $P$ are also
        the eigenvectors of the much smaller Gram matrix

        $$
        G = P P^\dagger
        $$

        Since the number of incoherent probe modes is typically much smaller than
        the number of probe pixels, $G$ has shape
        `(nModes, nModes)` and can be factorized much more efficiently than the
        full probe matrix. The orthogonal modes are then obtained from

        $$
        U^\dagger P
        $$

        which is equivalent to $S V^\dagger$.

        The batched implementation performs this operation for multiple scan
        positions simultaneously and processes the probe stack in chunks to limit
        temporary memory usage.
        """
        if self.params.OPR_fast_orthogonalization:
            return self._orthogonalizeIncoherentModes_batched()

        nFrames = self.experimentalData.numFrames
        n = self.reconstruction.Np
        nModes = self.reconstruction.probe_stack.shape[2]
        for pos in range(nFrames):
            probe = self.reconstruction.probe_stack[0, 0, :, 0, :, :, pos]
            probe = probe.reshape(nModes, n * n)

            U, s, Vh = self.svd(probe)

            modes = (s[:, None] * Vh).reshape(nModes, n, n)
            self.reconstruction.probe_stack[0, 0, :, 0, :, :, pos] = modes

    def _orthogonalizeIncoherentModes_batched(self):
        """Batched Gram-matrix equivalent of :meth:`orthogonalizeIncoherentModes`.

        For each frame the loop above computes ``s[:, None] * Vh`` from the SVD
        of a ``(nModes, Np**2)`` matrix P. Since ``P = U S Vh``, that product is
        just ``U^H P``, and U is the left singular matrix of the *tiny*
        ``(nModes, nModes)`` Gram matrix ``P P^H``. So the whole thing reduces to
        a batched factorization of a handful of 4x4 matrices plus one batched
        matmul -- no large SVD, and one kernel launch per chunk instead of one
        per frame.

        Measured 4.1x faster than the loop at 364 px / 202 frames / 4 modes, and
        1.8x at 512 px / 890 frames / 6 modes. The advantage *shrinks* with size:
        the loop's cost is dominated by per-frame launch overhead at small sizes,
        which is exactly what batching removes, while at large sizes the
        factorization itself dominates and batching has less to hide.

        Caveat: eigenvectors are only defined up to a per-mode phase, and when
        two modes carry near-equal power the vectors within that subspace are
        not determined at all. The mode *powers* (singular values) and the
        spanned subspace are reproduced exactly; individual mode vectors may
        differ from LAPACK's arbitrary choice. Guarded by
        ``params.OPR_fast_orthogonalization``.
        """
        stack = self.reconstruction.probe_stack
        xp = getArrayModule(stack)
        n = self.reconstruction.Np
        nModes = stack.shape[2]
        nFrames = stack.shape[-1]

        # Transposing the whole stack at once would allocate a second (and
        # third) copy of it -- 2.4 GB for a 364 px / 202 frame / 4 mode run, on
        # top of the stack itself. Work in frame chunks so the extra allocation
        # stays bounded regardless of frame count; the batched call is already
        # wide enough at a few dozen frames to hide launch overhead.
        elements_per_frame = nModes * n * n
        chunk = int(max(1, self._orthogonalization_chunk_bytes //
                        (elements_per_frame * stack.dtype.itemsize)))

        flat = stack[0, 0, :, 0, :, :, :].reshape(nModes, n * n, nFrames)
        for start in range(0, nFrames, chunk):
            stop = min(start + chunk, nFrames)
            # (nModes, Np**2, chunk) -> (chunk, nModes, Np**2)
            P = xp.ascontiguousarray(xp.moveaxis(flat[:, :, start:stop], 2, 0))
            G = P @ P.conj().transpose(0, 2, 1)
            # batched SVD of the tiny Hermitian Gram matrices; see gram_tsvd for
            # why this is used in preference to eigh. Already ordered by
            # descending mode power.
            U, _w, _Vh = xp.linalg.svd(G)
            modes = U.conj().transpose(0, 2, 1) @ P
            flat[:, :, start:stop] = xp.moveaxis(modes, 0, 2)
            del P, G, U, modes

    def average(self, arr):
        """
        Smooth a one-dimensional array by averaging neighboring values.

        Each interior element is replaced by the average of itself and its two
        nearest neighbors,

        $$
        a_j^{\mathrm{new}} = \frac{a_{j-1} + a_j + a_{j+1}}{3}
        $$

        while the first and last elements are averaged only with their single
        available neighbor.

        In OPR, this operation is used to smooth the position-dependent subspace
        coefficients when `params.OPR_neighbor_constraint` is enabled, thereby
        suppressing abrupt probe variations between successive scan positions.

        Args:
            arr (array-like):
                One-dimensional array of position-dependent coefficients.

        Returns:
            array-like:
                Smoothed array with the same shape as the input.
        
        Notes:
            This implementation currently operates on CuPy arrays and therefore
            requires GPU support.
        """

        arr_start = arr[:-1]
        arr_end = arr[1:]
        arr_end = cp.append(arr_end, 0)
        arr_start = cp.append(0, arr_start)
        divider = cp.ones_like(arr) * 3
        divider[0] = 2
        divider[-1] = 2
        return (arr + arr_end + arr_start) / divider

    def svd(self, P):
        r"""
        Compute the reduced singular-value decomposition of a matrix.

        The decomposition

        $$
        P = U S V^\dagger
        $$

        is evaluated with CuPy when `P` is stored on the GPU and with NumPy
        otherwise. In both cases, `full_matrices=False` is used to return the
        reduced SVD.

        Args:
            P (array-like):
                Input matrix to decompose.

        Returns:
            tuple:
                `(U, s, Vh)`, where `U` and `Vh` contain the left and right
                singular vectors and `s` contains the singular values.
        """
        if isGpuArray(P):
            try:
                return cp.linalg.svd(P, full_matrices=False)
            except:
                print(
                    "Something is wrong with SVD on cuda. Probably an installation error"
                )
                raise
        A, v, At = np.linalg.svd(asNumpyArray(P), full_matrices=False)
        if isGpuArray(P):
            A = cp.array(A)
            v = cp.array(v)
            At = cp.array(At)
        return A, v, At

    def rsvd(self, P, n_dim):
        """
        Compute a randomized truncated singular-value decomposition.

        This method delegates to `PtyLab.utils.fsvd.rsvd()` and returns a
        low-rank approximation of the input matrix using the requested subspace
        dimension.

        Args:
            P (array-like):
                Input matrix to decompose.

            n_dim (int):
                Number of singular components to retain.

        Returns:
            tuple:
                `(U, s, Vh)`, containing the truncated left singular vectors,
                singular values, and right singular vectors.
        """
        return rsvd(P, n_dim)
        # A, v, At = self.svd(P)
        # v[n_dim:] = 0
        # return A, v, At

    @staticmethod
    def gram_tsvd(A, n_dim):
        """
        Compute a truncated SVD through the Gram matrix.

        For a tall matrix $A$, the method forms the smaller Gram matrix

        $$
        G = A^\dagger A
        $$

        and uses

        $$
        A^\dagger A = V S^2 V^\dagger
        $$

        to recover the truncated singular-value decomposition of `A`. The retained
        left singular vectors are reconstructed from

        $$
        U = A V S^{-1}
        $$

        using the first `n_dim` components.

        The Gram matrix is formed in double precision for improved numerical
        stability, and division by numerically zero singular values is guarded.

        Args:
            A (array-like):
                Input matrix.

            n_dim (int):
                Number of singular components to retain.

        Returns:
            tuple:
                `(U, s, Vh)` containing the truncated singular-value decomposition.
        """
        xp = getArrayModule(A)
        n_dim = int(min(n_dim, A.shape[1]))

        G = (A.conj().T @ A).astype(xp.complex128)
        # G is Hermitian positive semi-definite, so its SVD and its
        # eigendecomposition coincide: the left singular vectors are the
        # eigenvectors and the singular values are the eigenvalues, already in
        # descending order. We use svd rather than eigh because eigh routes
        # through cupyx.cusolver, which is not importable in every CuPy/CUDA
        # installation (it needs libcusolver at a version cupy-cuda12x does not
        # always ship), whereas svd works through cupy's own bindings.
        V, w, _Vh = xp.linalg.svd(G)
        w = w[:n_dim]
        V = V[:, :n_dim]

        s = xp.sqrt(xp.clip(w.real, 0.0, None))
        V = V.astype(A.dtype)
        # guard the division for numerically-zero singular values
        s_safe = xp.where(s > 0, s, 1.0)
        U = (A @ V) / s_safe.astype(A.real.dtype)[None, :]
        return U, s.astype(A.real.dtype), V.conj().T

    def orthogonalizeProbeStack(self, probe_stack, n_dim):
        r"""
        Project the position-dependent probe stack onto a low-dimensional subspace.

        For each probe mode selected by `params.OPR_modes`, the probes from all
        scan positions are reshaped into a matrix

        $$
        A \in \mathbb{C}^{N_p^2 \times N_{\mathrm{frames}}}
        $$

        whose columns contain the flattened probe estimates at individual scan
        positions.

        A truncated singular-value decomposition is then computed using the method
        selected by `params.OPR_tsvd_type`. Only `n_dim` dominant components are
        retained, yielding a low-rank approximation of the position-dependent
        probe stack.

        If `params.OPR_neighbor_constraint` is enabled, the position-dependent
        subspace coefficients are smoothed over neighboring scan positions before
        the probe stack is reconstructed.

        The low-rank approximation is blended with the current probe stack using

        $$
        P^{\mathrm{new}} = \alpha P^{\mathrm{old}} + (1-\alpha)P^{\mathrm{low-rank}}
        $$

        where `alpha = params.OPR_alpha`.

        Args:
            probe_stack (array-like):
                Position-dependent probe stack.

            n_dim (int):
                Number of OPR subspace components to retain.

        Returns:
            array-like:
                Regularized probe stack with the same shape as the input.
        """
        xp = getArrayModule(probe_stack)
        n = self.reconstruction.Np
        nFrames = self.experimentalData.numFrames

        for i, mode in enumerate(self.OPR_modes):
            A = probe_stack[:, :, i, :, :, :].reshape(n * n, nFrames)

            if self.params.OPR_tsvd_type == "randomized":
                U, s, Vh = self.rsvd(A, n_dim)
            elif self.params.OPR_tsvd_type == "gram":
                U, s, Vh = self.gram_tsvd(A, n_dim)
            elif self.params.OPR_tsvd_type == "numpy":
                U, s, Vh = xp.linalg.svd(A, full_matrices=False)
                s = s.copy()
                s[n_dim:] = 0
            else:
                raise ValueError(
                    f"unknown OPR_tsvd_type {self.params.OPR_tsvd_type!r}; "
                    f"expected 'numpy', 'gram' or 'randomized'"
                )

            if self.params.OPR_neighbor_constraint:
                # Calculate the average of neigboring singular values
                content = s[:, None] * Vh
                for j in range(min(n_dim, content.shape[0])):
                    content[j] = self.average(content[j])

                probe_stack[:, :, i, :, :, :] = self.alpha * probe_stack[
                    :, :, i, :, :, :
                ] + (1 - self.alpha) * (U @ content).reshape(n, n, nFrames)
            else:
                update = (U @ (s[:, None] * Vh)).reshape(n, n, nFrames)
                probe_stack[:, :, i, :, :, :] *= self.alpha
                probe_stack[:, :, i, :, :, :] += (1 - self.alpha) * update

        return probe_stack

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Update the object patch using the ePIE correction.

        The object update uses the current probe and exit-wave correction `DELTA`
        with normalization by the maximum probe intensity.

        Args:
            objectPatch (np.ndarray):
                Current object patch at the scan position.

            DELTA (np.ndarray):
                Exit-wave correction after the intensity projection.

        Returns:
            np.ndarray:
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

    def probeUpdate(
        self, objectPatch: np.ndarray, DELTA: np.ndarray, weight: float, gimmel=0.1
    ):
        """
        Update the probe using the ePIE correction.

        The probe update is computed from the current object patch and exit-wave
        correction `DELTA`. The normalization includes the regularization term
        `gimmel` to avoid excessively large updates when the object intensity is
        small.

        The update can additionally be scaled by `weight`. In the current OPR
        reconstruction workflow, `weight` is set to 1.

        Args:
            objectPatch (np.ndarray):
                Current object patch at the scan position.

            DELTA (np.ndarray):
                Exit-wave correction after the intensity projection.

            weight (float):
                Multiplicative weight applied to the probe update.

            gimmel (float, optional):
                Small positive regularization term added to the normalization
                denominator. Default is `0.1`.

        Returns:
            np.ndarray:
                Updated probe.
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        frac = objectPatch.conj() / (
            xp.max(xp.sum(xp.abs(objectPatch) ** 2, axis=(0, 1, 2, 3))) + gimmel
        )
        frac = frac * weight
        r = self.reconstruction.probe + self.betaProbe * xp.sum(
            frac * DELTA, axis=(0, 1, 3), keepdims=True
        )
        return r
