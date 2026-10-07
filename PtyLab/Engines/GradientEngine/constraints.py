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


def _optimizer_state(engine, parameter):
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None:
        return {}
    return optimizer.state.get(parameter, {})


def _moments(engine, parameter):
    """Return optimizer state shaped like `parameter` (Adam moments, SGD momentum)."""
    return {
        key: value
        for key, value in _optimizer_state(engine, parameter).items()
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

        **schedule:
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If `threshold` is not a nonnegative number.

    Notes:
        The roll is cyclic, so it assumes the probe is small at the window
        edge.
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

        **schedule:
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If both bounds are None, a bound is negative, or `minimum` exceeds
            `maximum`.
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

        **schedule:
            `start`, `step`, `end` and `relax`, as in `Constraint`.

    Raises:
        ValueError:
            If neither `ramp` nor `global_phase` is selected.
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
        return model.gradient_preconditioners(indices)["object"]

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
            k_row = float(
                torch.angle(
                    (
                        weight[..., 1:, :] * obj[..., 1:, :] * obj[..., :-1, :].conj()
                    ).sum()
                )
            )
            k_col = float(
                torch.angle(
                    (
                        weight[..., :, 1:] * obj[..., :, 1:] * obj[..., :, :-1].conj()
                    ).sum()
                )
            )
        phi = 0.0
        if self.global_phase:
            phi = float(
                torch.angle((weight * obj * self._ramp(obj, k_row, k_col).conj()).sum())
            )
        return k_row, k_col, phi

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
