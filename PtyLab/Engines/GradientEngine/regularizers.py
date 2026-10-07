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
    the complex object $O$. This quadratic smoothness penalty suppresses strong
    gradients and smooths a noisy reconstruction.

    Args:
        model (PtychographyModel):
            Model whose `object` is penalized.

        weight (float, optional):
            Weight $w$ of the penalty. Defaults to 1e-4.

    Returns:
        torch.Tensor:
            Real scalar penalty.
    """
    obj = model.object
    dy = obj[..., 1:, :] - obj[..., :-1, :]
    dx = obj[..., :, 1:] - obj[..., :, :-1]
    return weight * (dy.abs().square().mean() + dx.abs().square().mean())
