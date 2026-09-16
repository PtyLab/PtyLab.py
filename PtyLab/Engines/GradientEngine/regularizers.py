"""Optional penalties on complex object/probe tensors, applied once per iteration.

Each callable takes (object_tensor, probe_tensor) and returns a real scalar tensor.
Use functools.partial to set a weight, or provide your own function with that
signature. All computations must stay in Torch to retain gradients.
"""


def object_smoothness(obj, probe, weight=1e-4):
    """Penalize neighboring complex object differences (amplitude and phase).

    R = weight * (mean(|D_y O|^2) + mean(|D_x O|^2))

    O is obj; D_y and D_x are forward differences along its last two axes.
    Each mean includes all leading dimensions and valid neighboring pairs;
    no differences wrap around the boundary. Both spatial axes need size >= 2.
    probe is unused and retained for the engine's regularizer interface.
    Returns a real scalar tensor with gradients through obj.
    This is a squared finite-difference penalty, not total variation.
    """
    # D_y O[y, x] = O[y + 1, x] - O[y, x]; D_x is analogous.
    dy = obj[..., 1:, :] - obj[..., :-1, :]
    dx = obj[..., :, 1:] - obj[..., :, :-1]
    return weight * (dy.abs().square().mean() + dx.abs().square().mean())
