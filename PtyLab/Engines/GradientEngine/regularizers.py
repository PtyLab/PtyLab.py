"""Optional penalties on a :class:`PtychographyModel` state."""


def object_smoothness(model, weight=1e-4):
    """This the standard total variation (TV) regularizer that penalizes squared neighboring differences of
    the complex object. For a noisier reconstruction, TV can penalize the strong gradients in the image and smoothen
    the reconstruction
    """
    obj = model.object
    dy = obj[..., 1:, :] - obj[..., :-1, :]
    dx = obj[..., :, 1:] - obj[..., :, :-1]
    return weight * (dy.abs().square().mean() + dx.abs().square().mean())
