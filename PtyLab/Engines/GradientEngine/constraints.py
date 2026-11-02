r"""
In-place constraints applied after optimizer steps.

Assign a list to `engine.constraints`. After every optimizer step the engine
calls each constraint as `constraint(engine, iteration)` under
`torch.no_grad`. A constraint runs at iterations `start`, `start + step`, ...
before `end` (exclusive; None means no end). `iteration` counts from zero
within one `reconstruct()` call, so within one phase of `engine.run`.

The relaxation $\rho$ (`relax`) blends the result with the unconstrained value,

$$
x \leftarrow \rho\,x + (1 - \rho)\,C(x),
$$

so $\rho = 0$ applies the constraint $C$ fully and values towards 1 apply it
gently. Constraints that move a parameter, such as recentring, transform that
parameter's optimizer state the same way, so Adam's moments stay aligned with
the pixels they belong to.
"""

from numbers import Integral, Real

import torch

from .batching import split_indices
from .models import OPRProbe, frame_modes


class Constraint:
    r"""
    Schedule and relaxation shared by every constraint.

    Subclasses implement `apply(engine)`, which runs under `torch.no_grad`.

    Args:
        start (int, optional):
            First iteration at which the constraint runs. Defaults to 0.

        step (int, optional):
            Interval between runs, at least 1. Defaults to 1.

        end (int, optional):
            Iteration before which the constraint stops (exclusive). If None,
            it never stops.

        relax (float, optional):
            Relaxation $\rho$ in $[0, 1)$. Defaults to 0.0.

    Raises:
        ValueError:
            If the schedule or the relaxation is invalid.

    Examples:
        ```python
        import torch
        from PtyLab.Engines.GradientEngine import Constraint


        class ObjectPhaseClamp(Constraint):
            # Limit the object phase to [-limit, limit], keeping the amplitude.

            def __init__(self, limit=1.0, **schedule):
                super().__init__(**schedule)
                self.limit = limit

            def apply(self, engine):  # runs under torch.no_grad
                obj = engine.model.object
                phase = obj.angle().clamp(-self.limit, self.limit)
                target = torch.polar(obj.abs(), phase)
                obj.copy_(self.relax * obj + (1 - self.relax) * target)


        engine.constraints = [ObjectPhaseClamp(limit=1.0, start=100, step=5)]
        ```
    """

    def __init__(self, start=0, step=1, end=None, relax=0.0):
        for name, value, minimum in (("start", start, 0), ("step", step, 1)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if end is not None and (
            isinstance(end, bool) or not isinstance(end, Integral) or end <= start
        ):
            raise ValueError("end must be None or an integer > start.")
        if isinstance(relax, bool) or not isinstance(relax, Real) or not 0 <= relax < 1:
            raise ValueError("relax must lie in [0, 1).")
        self.start, self.step, self.end, self.relax = start, step, end, float(relax)

    def active(self, iteration):
        """
        Return whether the schedule applies the constraint at an iteration.

        Args:
            iteration (int):
                Iteration index within the current `reconstruct()` call.

        Returns:
            bool:
                True if the constraint runs at `iteration`.
        """
        if iteration < self.start or (self.end is not None and iteration >= self.end):
            return False
        return (iteration - self.start) % self.step == 0

    def __call__(self, engine, iteration):
        if self.active(iteration):
            with torch.no_grad():
                self.apply(engine)

    def apply(self, engine):
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{type(self).__name__}(start={self.start}, step={self.step}, "
            f"end={self.end}, relax={self.relax})"
        )


def _moments(engine, parameter):
    """Return optimizer state shaped like `parameter` (Adam moments, SGD momentum)."""
    optimizer = getattr(engine, "optimizer", None)
    state = {} if optimizer is None else optimizer.state.get(parameter, {})
    return {
        key: value
        for key, value in state.items()
        if torch.is_tensor(value) and value.shape == parameter.shape
    }


def roll_parameter(engine, parameter, shifts):
    """
    Roll a parameter and its optimizer state over the last two axes.

    Args:
        engine (GradientEngine):
            Engine whose optimizer holds the state.

        parameter (torch.Tensor):
            Parameter rolled in place.

        shifts (tuple):
            `(rows, columns)` shift in pixels.
    """
    parameter.copy_(torch.roll(parameter, shifts, dims=(-2, -1)))
    for value in _moments(engine, parameter).values():
        value.copy_(torch.roll(value, shifts, dims=(-2, -1)))


def multiply_parameter(engine, parameter, factor):
    """
    Multiply a complex parameter and its optimizer state by a unit-modulus factor.

    First moments (Adam `exp_avg`, SGD `momentum_buffer`) rotate with the
    parameter exactly. Adam's `exp_avg_sq` holds separate real and imaginary
    second moments, which a rotation mixes; both are replaced by their mean,
    the rotation-invariant part.

    Args:
        engine (GradientEngine):
            Engine whose optimizer holds the state.

        parameter (torch.Tensor):
            Complex parameter multiplied in place.

        factor (torch.Tensor or complex):
            Unit-modulus factor, broadcastable to `parameter`.
    """
    parameter.mul_(factor)
    for key, value in _moments(engine, parameter).items():
        if key in ("exp_avg_sq", "max_exp_avg_sq") and value.is_complex():
            mean = 0.5 * (value.real + value.imag)
            value.copy_(torch.complex(mean, mean))
        else:
            value.mul_(factor)


def snapshot_basis(states):
    r"""
    Return the snapshot orthogonalization of a set of states.

    For states $S_m$ (rows of `states`) it diagonalizes the Gram matrix

    $$
    G_{mn} = \sum_{\mathbf{r}} S_m^*(\mathbf{r})\,S_n(\mathbf{r}) = \sum_k W_{mk}\,\lambda_k\,W^*_{nk},
    $$

    sorts $\lambda_1 \geq \lambda_2 \geq \ldots$ and returns $V = W^T$, as
    `PtyLab.utils.utils.orthogonalizeModes(..., method="snapShots")`. The states
    $S'_n = \sum_m V_{nm} S_m$ are orthogonal with powers $\lambda_n$.

    Args:
        states (torch.Tensor):
            Complex tensor of shape `(..., n, pixels)`; leading axes are
            batched.

    Returns:
        tuple:
            `(V, eigenvalues)` of shapes `(..., n, n)` and `(..., n)`.
    """
    # G = conj(S) S^T as in orthogonalizeModes; eigh in double precision, since weak
    # states have eigenvalues many orders below the strongest.
    gram = (states.conj() @ states.transpose(-2, -1)).to(torch.complex128)
    eigenvalues, W = torch.linalg.eigh(gram)
    order = eigenvalues.argsort(dim=-1, descending=True)  # strongest first
    eigenvalues = eigenvalues.gather(-1, order)
    W = W.gather(-1, order.unsqueeze(-2).expand_as(W))
    # V = W^T: row n holds the weights of the new state n.
    return W.transpose(-2, -1).to(states.dtype), eigenvalues


def mix_states(engine, parameter, dim, matrix):
    r"""
    Replace the states along one axis by linear combinations of them.

    $$
    S'_n = \sum_m M_{nm} S_m ,
    $$

    and the same for the optimizer state. First moments (Adam `exp_avg`)
    mix exactly like the states. Adam's second moments (`exp_avg_sq`, real
    and imaginary parts separately) cannot be mixed linearly; they become
    $v'_n = \sum_m |M_{nm}|^2 v_m$, which is exact if the gradients of
    different states are uncorrelated.

    Args:
        engine (GradientEngine):
            Engine whose optimizer holds the state.

        parameter (torch.Tensor):
            Parameter changed in place, for example the probe
            `(nlambda, 1, npsm, 1, Np, Np)` with `dim=2` or the object
            `(nlambda, nosm, 1, 1, No, No)` with `dim=1`.

        dim (int):
            State axis.

        matrix (torch.Tensor):
            Complex mixing matrix $M$ of shape `(*parameter.shape[:dim], n, n)`:
            one matrix per index of the axes before `dim`.
    """

    def mix(tensor, weights):
        # (*batch, n, ...) -> (*batch, n, rest): mix rows, write back in place.
        flat = tensor.reshape(*tensor.shape[: dim + 1], -1)
        tensor.copy_((weights @ flat).reshape(tensor.shape))

    mix(parameter, matrix)
    for key, value in _moments(engine, parameter).items():
        if key in ("exp_avg_sq", "max_exp_avg_sq"):
            # Real weights |M|^2 act on the real and imaginary parts alike.
            mix(value, matrix.abs().square().to(value.dtype))
        else:
            mix(value, matrix.to(value.dtype))


class ProbeCentering(Constraint):
    r"""
    Keep the probe's centre of mass at the window centre.

    The probe intensity $\sum |P|^2$ over all probe axes gives a centre of mass
    $\mathbf{c}$ in pixels from the window centre. When
    $\max(|c_{\mathrm{row}}|, |c_{\mathrm{col}}|) \geq$ `threshold`, the probe,
    the object and their optimizer state are rolled by

    $$
    -\mathrm{round}\left((1 - \rho)\,\mathbf{c}\right).
    $$

    Exit waves then move by a whole number of pixels inside the window, which
    leaves far-field (Fraunhofer) intensities unchanged; for near-field
    propagators recentring changes the prediction.

    Args:
        threshold (float, optional):
            Offset in pixels below which nothing is done. Defaults to 2.

        **schedule (dict):
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If `threshold` is not a nonnegative number.

    Notes:
        The roll is cyclic, so it assumes the probe is small at the window
        edge.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import ProbeCentering

        engine.constraints = [ProbeCentering(threshold=2)]
        ```
    """

    def __init__(self, threshold=2, **schedule):
        super().__init__(**schedule)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or threshold < 0
        ):
            raise ValueError("threshold must be a nonnegative number of pixels.")
        self.threshold = threshold

    @staticmethod
    def centre_of_mass(field):
        r"""
        Return the centre of mass of $\sum |\mathrm{field}|^2$.

        Args:
            field (torch.Tensor):
                Complex field with spatial axes last; all leading axes are
                summed.

        Returns:
            tuple:
                `(row, column)` in pixels relative to the window centre.
        """
        intensity = (
            field.detach().abs().square().reshape(-1, *field.shape[-2:]).sum(dim=0)
        )
        height, width = intensity.shape
        rows = torch.arange(height, device=field.device) - height // 2
        cols = torch.arange(width, device=field.device) - width // 2
        total = intensity.sum()
        return (
            float((intensity.sum(-1) * rows).sum() / total),
            float((intensity.sum(-2) * cols).sum() / total),
        )

    def apply(self, engine):
        model = engine.model
        row, col = self.centre_of_mass(model.probe.field)
        if max(abs(row), abs(col)) < self.threshold:
            return
        shifts = (
            -int(round((1 - self.relax) * row)),
            -int(round((1 - self.relax) * col)),
        )
        if shifts == (0, 0):
            return
        roll_parameter(engine, model.probe.field, shifts)
        roll_parameter(engine, model.object, shifts)


class ObjectAmplitudeClamp(Constraint):
    """
    Limit the object amplitude to `[minimum, maximum]`, keeping its phase.

    Optimizer state is unchanged: this is a projection, not a move.

    Args:
        maximum (float, optional):
            Upper amplitude bound, or None. `1` suits a transmission sample
            without gain. Defaults to 1.0.

        minimum (float, optional):
            Lower amplitude bound, or None. Defaults to None.

        **schedule (dict):
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If both bounds are None, a bound is negative, or `minimum` exceeds
            `maximum`.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import ObjectAmplitudeClamp

        engine.constraints = [ObjectAmplitudeClamp(maximum=1.0)]  # no gain
        ```
    """

    def __init__(self, maximum=1.0, minimum=None, **schedule):
        super().__init__(**schedule)
        if maximum is None and minimum is None:
            raise ValueError("Give at least one of minimum and maximum.")
        for value in (maximum, minimum):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or value < 0
            ):
                raise ValueError("Amplitude bounds must be nonnegative numbers.")
        if maximum is not None and minimum is not None and minimum > maximum:
            raise ValueError("minimum must not exceed maximum.")
        self.maximum, self.minimum = maximum, minimum

    def apply(self, engine):
        obj = engine.model.object
        amplitude = obj.abs()
        clamped = amplitude.clamp(min=self.minimum, max=self.maximum)
        # The phase of a zero pixel is undefined; raise it along the real axis.
        phase = torch.where(amplitude > 0, obj / amplitude.clamp_min(1e-30), 1)
        target = clamped * phase
        obj.copy_(self.relax * obj + (1 - self.relax) * target)


class AbsorbingObject(Constraint):
    r"""
    Pull the object towards a pure-absorption object, as `Params.absObjectSwitch`.

    Each call blends the object with its modulus,

    $$
    O \leftarrow (1 - \beta)\,O + \beta\,|O|,
    $$

    with $\beta$ = `beta`. It shrinks the phase of every pixel towards 0
    and leaves the amplitude of real positive pixels unchanged. It suits
    samples whose phase carries little information, and damps phase noise in
    weakly lit pixels. Called once per iteration it matches the classic
    engines' constraint; with `updatePerBatch` it runs after every step.

    Optimizer state is unchanged: this is a small pull, not a move of the
    parameter. With relaxation $\rho$ the pull is $(1 - \rho)\beta$.

    Args:
        beta (float, optional):
            Pull $\beta$ in $(0, 1]$ per call, as `Params.absObjectBeta`.
            Defaults to 0.01.

        **schedule (dict):
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If `beta` is not in $(0, 1]$.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import AbsorbingObject

        engine.constraints = [AbsorbingObject(beta=0.01, end=200)]  # first 200 only
        ```
    """

    def __init__(self, beta=0.01, **schedule):
        super().__init__(**schedule)
        if isinstance(beta, bool) or not isinstance(beta, Real) or not 0 < beta <= 1:
            raise ValueError("beta must lie in (0, 1].")
        self.beta = float(beta)

    def apply(self, engine):
        obj = engine.model.object
        beta = (1 - self.relax) * self.beta  # relaxation keeps a fraction of O
        modulus = obj.abs()  # before the in-place scaling below
        obj.mul_(1 - beta).add_(modulus, alpha=beta)


class PhaseRampRemoval(Constraint):
    r"""
    Move a linear phase ramp and a global phase from the object to the probe.

    For

    $$
    O'(\mathbf{x}) = O(\mathbf{x})\,e^{-i(\mathbf{k}\cdot\mathbf{x} + \phi)}, \qquad P'(\mathbf{r}) = P(\mathbf{r})\,e^{i(\mathbf{k}\cdot\mathbf{r} + \phi)},
    $$

    each exit wave changes only by the constant phase
    $e^{-i\mathbf{k}\cdot\mathbf{x}_j}$ of its scan position, so every
    predicted intensity is unchanged, for any propagator. This fixes a gauge
    freedom of blind ptychography instead of changing the fit.

    $\mathbf{k}$ is estimated from the illumination-weighted phase of
    neighbouring object pixels, which does not need phase unwrapping:

    $$
    k_{\mathrm{row}} = \arg\sum_{\mathbf{x}} w(\mathbf{x})\, O(\mathbf{x} + \mathbf{e}_{\mathrm{row}})\,O^*(\mathbf{x}),
    $$

    and likewise along columns. $\phi$ is the weighted mean phase after the
    ramp is removed. The relaxation $\rho$ removes only the fraction
    $1 - \rho$ of both.

    Args:
        ramp (bool, optional):
            Remove the linear phase ramp. Defaults to True.

        global_phase (bool, optional):
            Remove the global phase. Defaults to True.

        **schedule (dict):
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If neither `ramp` nor `global_phase` is selected.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import PhaseRampRemoval, ProbeCentering

        engine.constraints = [ProbeCentering(), PhaseRampRemoval(step=10)]
        ```
    """

    def __init__(self, ramp=True, global_phase=True, **schedule):
        super().__init__(**schedule)
        if not (ramp or global_phase):
            raise ValueError("Select ramp, global_phase or both.")
        self.ramp, self.global_phase = bool(ramp), bool(global_phase)

    @staticmethod
    def weights(engine):
        r"""
        Return the illumination over the whole scan.

        $$
        w(\mathbf{x}) = \sum_j |P(\mathbf{x} - \mathbf{x}_j)|^2
        $$

        Args:
            engine (GradientEngine):
                Engine with a prepared model.

        Returns:
            torch.Tensor:
                Real weights with the shape of the object.
        """
        model = engine.model
        indices = torch.arange(len(model.positions), device=model.object.device)
        # Batch by batch, so the engine's batch size bounds the memory; the
        # size is None (one batch) before the first run.
        size = getattr(engine, "effectiveBatchSize", None)
        weight = sum(
            model.illumination(batch) for batch in split_indices(indices, size)
        )
        return weight.expand(model.object.shape)

    def estimate(self, engine):
        r"""
        Estimate the phase ramp and the global phase of the object.

        Args:
            engine (GradientEngine):
                Engine with a prepared model.

        Returns:
            tuple:
                `(k_row, k_col, phi)`: $k_{\mathrm{row}}$ and $k_{\mathrm{col}}$
                in radians per pixel and $\phi$ in radians. Disabled parts are
                zero.
        """
        obj = engine.model.object
        weight = self.weights(engine).to(obj.real.dtype)
        k_row = k_col = 0.0
        if self.ramp:
            k_row = self._neighbour_phase(weight, obj, dim=-2)
            k_col = self._neighbour_phase(weight, obj, dim=-1)
        phi = 0.0
        if self.global_phase:
            phi = float(
                torch.angle((weight * obj * self._ramp(obj, k_row, k_col).conj()).sum())
            )
        return k_row, k_col, phi

    @staticmethod
    def _neighbour_phase(weight, field, dim):
        r"""
        Return $\arg\sum_{\mathbf{x}} w(\mathbf{x} + \mathbf{e})\, O(\mathbf{x} + \mathbf{e})\,O^*(\mathbf{x})$ along `dim`.

        This is the ramp $k$ in radians per pixel along that axis.
        """
        length = field.shape[dim] - 1
        later, earlier = field.narrow(dim, 1, length), field.narrow(dim, 0, length)
        return float(
            torch.angle((weight.narrow(dim, 1, length) * later * earlier.conj()).sum())
        )

    @staticmethod
    def _ramp(field, k_row, k_col):
        r"""
        Return the phase ramp $e^{i(k_{\mathrm{row}} r + k_{\mathrm{col}} c)}$.

        $r$ and $c$ are measured in pixels from the window centre.
        """
        height, width = field.shape[-2:]
        rows = (
            torch.arange(height, device=field.device, dtype=field.real.dtype)
            - height // 2
        )
        cols = (
            torch.arange(width, device=field.device, dtype=field.real.dtype)
            - width // 2
        )
        phase = k_row * rows[:, None] + k_col * cols[None, :]
        return torch.polar(torch.ones_like(phase), phase).to(field.dtype)

    def apply(self, engine):
        model = engine.model
        k_row, k_col, phi = (
            (1 - self.relax) * value for value in self.estimate(engine)
        )
        obj, probe = model.object, model.probe.field
        object_factor = self._ramp(obj, k_row, k_col).conj() * complex(
            torch.polar(torch.tensor(1.0), torch.tensor(-phi))
        )
        # The object ramp is centred on the object; at probe pixel r of frame j
        # the object coordinate is r + x_j relative to the object's corner, so
        # the probe ramp in window coordinates differs only by a per-frame
        # constant, which intensities do not see.
        probe_factor = self._ramp(probe, k_row, k_col) * complex(
            torch.polar(torch.tensor(1.0), torch.tensor(phi))
        )
        multiply_parameter(engine, obj, object_factor)
        multiply_parameter(engine, probe, probe_factor)


def _orthogonalize_states(engine, parameter, dim, relax):
    """Rotate the states on axis `dim` of `parameter` into orthogonal ones.

    Returns the purities, one per index of the axes before `dim`, or None for
    a single state.
    """
    count = parameter.shape[dim]
    if count < 2:
        return None
    # (*batch, n, ...) -> (*batch, n, pixels); one basis per batch index.
    V, eigenvalues = snapshot_basis(parameter.reshape(*parameter.shape[: dim + 1], -1))
    identity = torch.eye(count, dtype=V.dtype, device=V.device)
    mix_states(engine, parameter, dim, relax * identity + (1 - relax) * V)
    # Purity sqrt(sum w_k^2), w_k = lambda_k / sum(lambda), as in
    # BaseEngine.orthogonalization.
    weights = eigenvalues.clamp(min=0)
    weights = weights / weights.sum(-1, keepdim=True)
    return weights.square().sum(-1).sqrt().flatten().tolist()


class ProbeModeOrthogonalization(Constraint):
    r"""
    Rotate the probe modes into orthogonal modes, strongest first.

    This is the snapshot orthogonalization (`orthogonalizeModes` with
    `method="snapShots"`, used by mPIE's `orthogonalizationSwitch`), computed
    in Torch on the engine's device (see `snapshot_basis`). It diagonalizes
    the Gram matrix of the modes of each wavelength,

    $$
    G_{mn} = \sum_{\mathbf{r}} P_m^*(\mathbf{r})\,P_n(\mathbf{r}) = \sum_k W_{mk}\,\lambda_k\,W^*_{nk},
    $$

    with eigenvalues sorted as $\lambda_1 \geq \lambda_2 \geq \ldots$, and
    takes $V = W^T$ as `orthogonalizeModes` does. The new modes $P'_n = \sum_m V_{nm} P_m$ are
    orthogonal, $\sum_{\mathbf{r}} P'^*_m P'_n = \lambda_n\,\delta_{mn}$, so
    their powers are the eigenvalues of $G$. A unitary
    rotation of the modes leaves every predicted intensity unchanged, so this
    only picks a basis: it does not change the fit.

    With relaxation $\rho$ the modes become $\rho P + (1 - \rho)P'$, that is
    the mixing matrix $\rho I + (1 - \rho) V$, which is no longer unitary.
    The optimizer state is mixed with the same matrix (see `mix_states`).
    For an `OPRProbe` the modes of every frame are orthogonalized separately,
    like `params.OPR_orthogonalize_modes` in the classic `OPR` engine.
    It takes the schedule arguments `start`, `step`, `end` and `relax` of
    `Constraint`.

    Notes:
        Each eigenvector is fixed only up to a phase, so the result agrees
        with `orthogonalizeModes` up to one phase factor per mode, which no intensity
        sees.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import ProbeModeOrthogonalization

        reconstruction.npsm = 4
        reconstruction.initializeObjectProbe()
        engine = GradientEngine(reconstruction, experimentalData, params, monitor)
        engine.constraints = [ProbeModeOrthogonalization(step=10)]
        engine.reconstruct()
        engine.model.probe.mode_powers()  # (nlambda, npsm), strongest first
        ```
    """

    def apply(self, engine):
        # Shared (nlambda, 1, npsm, 1, Np, Np) or OPR stack with frames first:
        # the mode axis is -4 in both, so an OPR stack is orthogonalized per frame.
        field = engine.model.probe.field
        purities = _orthogonalize_states(engine, field, field.dim() - 4, self.relax)
        if purities:
            # One value per call, the mean over wavelengths (and frames), stored
            # as BaseEngine does; the monitor shows it.
            purity = sum(purities) / len(purities)
            engine.reconstruction.purityProbe = purity
            engine.reconstruction.purityProbeHist.append(purity)


class ObjectStateOrthogonalization(Constraint):
    r"""
    Rotate the object states into orthogonal states, strongest first.

    The object counterpart of `ProbeModeOrthogonalization`, as
    `BaseEngine.orthogonalization` does for `nosm > 1` in the classic engines. For the object states
    $O_a$ of each wavelength, over the whole object grid,

    $$
    G_{ab} = \sum_{\mathbf{x}} O_a^*(\mathbf{x})\,O_b(\mathbf{x}),\qquad
    O'_b = \sum_a V_{ba} O_a ,
    $$

    with $V = W^T$ from the eigenvectors of $G$, strongest first. The
    rotation is unitary over the object states only, so for every probe mode
    $m$ the sum $\sum_a |\mathcal{D}\{O_a P_m\}|^2$, and every prediction, is
    unchanged. It can therefore run together with
    `ProbeModeOrthogonalization` (`BaseEngine.orthogonalization` orthogonalizes
    the object only when there is one probe mode). The optimizer state is mixed as in
    `mix_states`; `relax` blends as for the probe modes.

    Notes:
        The Gram matrix also counts unlit object pixels, as
        `BaseEngine.orthogonalization` does;
        they hold the start value and change $\lambda_k$ but not the fit.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import (
            ObjectStateOrthogonalization,
            ProbeModeOrthogonalization,
        )

        engine.constraints = [
            ProbeModeOrthogonalization(step=10),
            ObjectStateOrthogonalization(step=10),  # needs nosm > 1
        ]
        ```
    """

    def apply(self, engine):
        obj = engine.model.object  # (nlambda, nosm, 1, 1, No, No)
        purities = _orthogonalize_states(engine, obj, 1, self.relax)
        for purity in purities or []:
            engine.reconstruction.purityObject = purity


class AbsorbingProbeBoundary(Constraint):
    r"""
    Damp the probe towards the edge of its window, akin to
    `Params.absorbingProbeBoundary`.

    Each call multiplies every probe mode by

    $$
    P \leftarrow \big[(1 - a) + a\,W(\mathbf{r})\big]\,P, \qquad
    W(\mathbf{r}) = \exp\left[-\left(\frac{|\mathbf{r}|^2}{2\sigma^2}\right)^{10}\right], \qquad
    \sigma = \frac{3}{4}\,\frac{N_p}{2.355},
    $$

    with $\mathbf{r}$ in pixels on the probe grid of `Reconstruction.xp`
    (`linspace(-Np/2, Np/2, Np)`) and $a$ = `aleph`. $W$ is the
    super-Gaussian window of `BaseEngine._probeWindow`: about 1 out to
    $0.4\,N_p$ from the centre and 0 beyond $0.5\,N_p$, so light near the
    window edges and corners decays by $1 - a$ per call while the beam is
    untouched.

    Probe light far from the beam reaches object pixels nothing else lights.
    Such a pair is free to fit diffuse intensity, so without this constraint
    a weak mode can drift to the window corners.

    The factor is real and positive per pixel, so Adam's first moment is
    scaled by it and the second moment by its square, which keeps the
    normalized step $m/\sqrt{v}$ of each pixel unchanged. With relaxation
    $\rho$ the factor is $\rho + (1 - \rho)\big[(1 - a) + a\,W\big]$.

    Args:
        aleph (float, optional):
            Damping $a$ in $(0, 1]$ per call, as
            `Params.absorbingProbeBoundaryAleph`. Defaults to 0.05.

        **schedule (dict):
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If `aleph` is not in $(0, 1]$.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import AbsorbingProbeBoundary

        engine.constraints = [AbsorbingProbeBoundary(aleph=0.05)]
        ```
    """

    def __init__(self, aleph=0.05, **schedule):
        super().__init__(**schedule)
        if isinstance(aleph, bool) or not isinstance(aleph, Real) or not 0 < aleph <= 1:
            raise ValueError("aleph must lie in (0, 1].")
        self.aleph = float(aleph)

    @staticmethod
    def window(size, device=None, dtype=torch.float32):
        r"""
        Return an absorbing-boundary window $W$ for an `size`^2 probe.

        Args:
            size (int):
                Probe pixels per side, $N_p$.

            device (torch.device or str, optional):
                Device of the result.

            dtype (torch.dtype, optional):
                Real dtype of the result. Defaults to `torch.float32`.

        Returns:
            torch.Tensor:
                $W$ of shape `(size, size)`.
        """
        # In pixels dxp cancels: |r|^2 / (2 sigma^2) with r on Reconstruction.xp's linspace grid.
        x = torch.linspace(
            -size / 2, size / 2, size, device=device, dtype=torch.float64
        )
        sigma = 3 / 4 * size / 2.355
        radius2 = x[:, None] ** 2 + x[None, :] ** 2
        return torch.exp(-((radius2 / (2 * sigma**2)) ** 10)).to(dtype)

    def apply(self, engine):
        field = engine.model.probe.field  # (nlambda, 1, npsm, 1, Np, Np)
        W = self.window(field.shape[-1], field.device, field.real.dtype)
        factor = self.relax + (1 - self.relax) * ((1 - self.aleph) + self.aleph * W)
        field.mul_(factor)
        for key, value in _moments(engine, field).items():
            # m scales like P; v (real and imaginary parts) like |P|^2.
            value.mul_(factor**2 if key in ("exp_avg_sq", "max_exp_avg_sq") else factor)


class OPRProjection(Constraint):
    r"""
    Keep a per-frame probe stack low rank (orthogonal probe relaxation).

    For each wavelength and probe mode, the frame probes are the rows of
    $A$ ($A_{j\mathbf{r}} = P_j(\mathbf{r})$, $J$ frames). With the
    eigenvectors $V$ of the small Gram matrix $G = AA^\dagger$ (see
    `frame_modes`), the closest rank-$K$ stack is

    $$
    A_K = V_K V_K^\dagger A = \sum_{k=1}^{K} v_k\,(\sigma_k u_k^\dagger),
    $$

    so every frame probe becomes a combination of $K$ shared patterns
    $\sigma_k u_k^\dagger$ with frame coefficients $v_{jk}$. With
    relaxation $\rho$ the stack becomes $\rho A + (1 - \rho) A_K$; $\rho$
    plays the role of `params.OPR_alpha` of the classic `OPR` engine (0.05
    there). It needs an `OPRProbe` (`GradientEngine.relaxProbe`).

    Adam's moments are left as they are: the projection removes the
    directions outside the subspace, and the next step starts from there.

    Args:
        rank (int, optional):
            Number $K$ of kept components, like `params.OPR_subspace`.
            Defaults to 4.

        **schedule (dict):
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If `rank` is not a positive integer.

    Examples:
        ```python
        from PtyLab.Engines.GradientEngine import OPRProjection

        engine.relaxProbe()  # one probe per frame, from the shared probe
        engine.constraints = [OPRProjection(rank=4, relax=0.05)]
        engine.reconstruct()
        ```
    """

    def __init__(self, rank=4, **schedule):
        super().__init__(**schedule)
        if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
            raise ValueError("rank must be a positive integer.")
        self.rank = rank

    def apply(self, engine):
        probe = engine.model.probe
        if not isinstance(probe, OPRProbe):
            raise TypeError(
                "OPRProjection needs an OPRProbe; call engine.relaxProbe()."
            )
        field = probe.field  # (J, nlambda, 1, npsm, 1, Np, Np)
        if self.rank >= field.shape[0]:
            return  # every stack of J frames has rank <= J
        # One wavelength and mode at a time keeps the extra memory to one mode.
        for wavelength in range(field.shape[1]):
            for mode in range(field.shape[3]):
                frames = field[:, wavelength, 0, mode, 0]  # (J, Np, Np), a view
                rows = frames.flatten(-2)  # A: (J, N)
                _, V = frame_modes(rows)
                V = V[:, : self.rank]  # (J, K), strongest first
                low_rank = V @ (V.conj().T @ rows)  # A_K = V_K V_K^† A
                # rho A + (1 - rho) A_K, in place to keep one extra copy of A.
                low_rank.mul_(1 - self.relax).add_(rows, alpha=self.relax)
                frames.copy_(low_rank.view(frames.shape))
