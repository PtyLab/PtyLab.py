import pytest
import numpy as np
from skimage import data

try:
    from PtyLab.ProbeEngines.StandardProbe import SHGProbe
    HAS_PROBEENGINES = True
except (ImportError, ValueError):
    HAS_PROBEENGINES = False

pytestmark = pytest.mark.skipif(not HAS_PROBEENGINES, reason="ProbeEngines not ready")


def test_shg_probe_convergence():
    # bundled with scikit-image (no download); same image as imageio's "camera.png"
    target = data.camera().astype(np.float32)
    target = target / np.linalg.norm(target)
    engine = SHGProbe()
    engine.probe = np.random.rand(*target.shape)

    for i in range(1000):
        current_estimate = engine.get(None)
        new_estimate = target
        engine.push(new_estimate, None, None)
