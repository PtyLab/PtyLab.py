from unittest import TestCase
import numpy as np
import imageio
from PtyLab.ProbeEngines.StandardProbe import SHGProbe


class TestSHGProbe(TestCase):
    target = imageio.imread("imageio:camera.png").astype(np.float32)
    target = target / np.linalg.norm(target)
    engine = SHGProbe()
    engine.probe = np.random.rand(*target.shape)

    for i in range(1000):

        current_estimate = engine.get(None)
        if i % 10 == 0:
            print(np.linalg.norm(current_estimate - target))
        new_estimate = target
        engine.push(new_estimate, None, None)
