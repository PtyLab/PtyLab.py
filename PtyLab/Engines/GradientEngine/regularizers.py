"""
Optional penalties on a `PtychographyModel` state.
"""


def object_smoothness(model, weight=1e-4):
    r"""
    Penalize squared differences between neighbouring object pixels.

    $$
    \mathcal{R} = w\left(\left\langle |O_{y+1,x} - O_{y,x}|^2 \right\rangle + \left\langle |O_{y,x+1} - O_{y,x}|^2 \right\rangle\right)
    $$

    where $\langle\cdot\rangle$ is the mean over all pixel pairs (and modes) of
    the complex object $O$. This quadratic smoothness penalty (an L2 penalty on
    the object gradient) suppresses strong gradients and smooths a noisy
    reconstruction, edges included. `object_tv` keeps edges sharper.

    Args:
        model (PtychographyModel):
            Model whose `object` is penalized.

        weight (float, optional):
            Weight $w$ of the penalty. Defaults to 1e-4.

    Returns:
        torch.Tensor:
            Real scalar penalty.

    Examples:
        ```python
        from functools import partial
        from PtyLab.Engines.GradientEngine.regularizers import object_smoothness

        engine.regularizers = [partial(object_smoothness, weight=1e-4)]
        ```
    """
    obj = model.object
    dy = obj[..., 1:, :] - obj[..., :-1, :]
    dx = obj[..., :, 1:] - obj[..., :, :-1]
    return weight * (dy.abs().square().mean() + dx.abs().square().mean())


def object_tv(model, weight=1e-4, epsilon=1e-6):
    r"""
    Penalize the isotropic total variation (TV) of the object.

    $$
    \mathcal{R} = w\left\langle \sqrt{|O_{y+1,x} - O_{y,x}|^2 + |O_{y,x+1} - O_{y,x}|^2 + \epsilon} \right\rangle
    $$

    where $\langle\cdot\rangle$ is the mean over all pixels (and modes) of the
    complex object $O$. Unlike `object_smoothness`, which squares the
    differences, TV grows only linearly with them: it removes small noisy
    variations but lets a few large jumps, such as sharp edges, stay. The
    offset $\epsilon$ keeps the gradient finite where the object is flat.

    Args:
        model (PtychographyModel):
            Model whose `object` is penalized.

        weight (float, optional):
            Weight $w$ of the penalty. Defaults to 1e-4.

        epsilon (float, optional):
            Offset $\epsilon$ under the square root, in squared object units.
            Differences much smaller than $\sqrt{\epsilon}$ are penalized
            quadratically, as in `object_smoothness`. Defaults to 1e-6.

    Returns:
        torch.Tensor:
            Real scalar penalty.

    Examples:
        ```python
        from functools import partial
        from PtyLab.Engines.GradientEngine.regularizers import object_tv

        engine.regularizers = [partial(object_tv, weight=1e-3)]
        ```
    """
    obj = model.object
    # Forward differences, cropped to the common (N - 1, N - 1) grid so that
    # dy and dx belong to the same pixel.
    dy = obj[..., 1:, :-1] - obj[..., :-1, :-1]
    dx = obj[..., :-1, 1:] - obj[..., :-1, :-1]
    return weight * (dy.abs().square() + dx.abs().square() + epsilon).sqrt().mean()
