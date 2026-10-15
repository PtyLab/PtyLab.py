"""
Reconstruction phases or ordered runs of optimizer settings for `GradientEngine.run`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .losses import amplitude_loss

_SETTINGS = {
    "parameters": "parameters",
    "optimizerType": "optimizerType",
    "loss": "lossFunction",
    "constraints": "constraints",
    "readoutSigma": "readoutSigma",
}


@dataclass
class Phase:
    """
    One stretch of a reconstruction with fixed optimizer settings.

    The phase runs with a fresh optimizer, starting from the estimates the
    previous phase left. Every optional field left at None keeps the engine's
    own setting.

    Args:
        iterations (int):
            Number of full-scan iterations.

        parameters (Mapping, optional):
            Parameter selection with learning rates, as `engine.parameters`.

        optimizerType (str, optional):
            `"adam"` or `"preconditioned"`, as `engine.optimizerType`.

        loss (callable, optional):
            Loss function, as `engine.lossFunction`.

        constraints (Sequence, optional):
            Constraints applied after every step, as `engine.constraints`.

        readoutSigma (float or array_like, optional):
            Readout noise handed to the loss, as `engine.readoutSigma`.

        name (str, optional):
            Label shown in progress output. Defaults to `""`.
    """

    iterations: int
    parameters: Optional[Mapping[str, Mapping[str, Any]]] = None
    optimizerType: Optional[str] = None
    loss: Optional[Callable] = None
    constraints: Optional[Sequence[Callable]] = None
    readoutSigma: Optional[Any] = None
    name: str = ""

    def settings(self):
        """
        Return the engine attribute values this phase sets.

        Returns:
            dict:
                Values keyed by engine attribute name, for the fields that are
                not None.
        """
        return {
            attribute: getattr(self, name)
            for name, attribute in _SETTINGS.items()
            if getattr(self, name) is not None
        }


def blind_recipe(
    iterations=1000,
    loss=None,
    start_fraction=0.2,
    object_lr=0.03,
    probe_lr=0.1,
    start_lr=0.3,
    constraints=None,
    read_out_sigma=None,
):
    r"""
    Build two phases that recover object and probe from a flat start.

    1. A fraction `start_fraction` of the iterations runs PIE-style
       preconditioned descent on the plain amplitude loss, with object and
       probe at `start_lr`, and without readout noise. This finds the right
       basin from a generic `circ` probe.
    2. Adam on `loss` runs for the rest, with object rate `object_lr` and
       scale-free probe rate `probe_lr`, and readout noise `read_out_sigma`.

    Args:
        iterations (int, optional):
            Total number of full-scan iterations. Defaults to 1000.

        loss (callable, optional):
            Loss of the Adam phase. If None, the engine's loss is kept.

        start_fraction (float, optional):
            Fraction of the iterations in the preconditioned phase, in
            $(0, 1)$. Defaults to 0.2.

        object_lr (float, optional):
            Adam learning rate of the object. Defaults to 0.03.

        probe_lr (float, optional):
            Scale-free Adam learning rate of the probe. Defaults to 0.1.

        start_lr (float, optional):
            Preconditioned step size of object and probe. Defaults to 0.3.

        constraints (Sequence, optional):
            Constraints of both phases. If None, the engine's are kept.

        read_out_sigma (float or array_like, optional):
            Readout noise $\sigma$ of the Adam phase, set as
            `engine.readoutSigma`: a scalar for every pixel, or one
            $(N_p, N_p)$ map in centered order, for example
            `dark.std(axis=0)` of dark frames, in the units of `loss`'s
            `read_out_sigma`. If None, the Adam phase keeps
            `engine.readoutSigma`.

    Notes:
        The start always runs with $\sigma = 0$. On measured data with the
        background in the model, a shifted amplitude start
        ($c = 3/8 + \sigma^2$) made the probe collapse in a following shifted
        Poisson phase; the plain start did not, and was as good or better on
        simulated data.

    Returns:
        list:
            The two `Phase` objects, for `GradientEngine.run`.

    Raises:
        ValueError:
            If `start_fraction` is outside $(0, 1)$ or `iterations` is too small
            for two phases.
    """
    if not 0 < start_fraction < 1:
        raise ValueError("start_fraction must lie in (0, 1).")
    start = max(1, round(start_fraction * iterations))
    if start >= iterations:
        raise ValueError("iterations is too small for two phases.")
    return [
        Phase(
            start,
            {"object": {"lr": start_lr}, "probe.field": {"lr": start_lr}},
            optimizerType="preconditioned",
            loss=amplitude_loss,
            constraints=constraints,
            # Plain amplitude has no readout term; 0 overrides engine.readoutSigma.
            readoutSigma=0.0,
            name="preconditioned amplitude start",
        ),
        Phase(
            iterations - start,
            {"object": {"lr": object_lr}, "probe.field": {"lr": probe_lr}},
            optimizerType="adam",
            loss=loss,
            constraints=constraints,
            readoutSigma=read_out_sigma,
            name="adam",
        ),
    ]
