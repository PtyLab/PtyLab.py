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

        name (str, optional):
            Label shown in progress output. Defaults to `""`.
    """

    iterations: int
    parameters: Optional[Mapping[str, Mapping[str, Any]]] = None
    optimizerType: Optional[str] = None
    loss: Optional[Callable] = None
    constraints: Optional[Sequence[Callable]] = None
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
):
    """
    Build two phases that recover object and probe from a flat start.

    1. A fraction `start_fraction` of the iterations runs PIE-style
       preconditioned descent on the amplitude loss, with object and probe at
       `start_lr`. This finds the right basin from a generic `circ` probe.
    2. Adam on `loss` runs for the rest, with object rate `object_lr` and
       scale-free probe rate `probe_lr`.

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
            name="preconditioned amplitude start",
        ),
        Phase(
            iterations - start,
            {"object": {"lr": object_lr}, "probe.field": {"lr": probe_lr}},
            optimizerType="adam",
            loss=loss,
            constraints=constraints,
            name="adam",
        ),
    ]
