"""Optional framework checks must run even when neither framework is installed."""

import os
import subprocess
import sys


def test_imports_without_tensorflow_or_torch():
    code = '''
import importlib.abc
import sys

class NoFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"tensorflow", "torch"}:
            raise ModuleNotFoundError(f"{root} intentionally unavailable", name=root)

sys.meta_path.insert(0, NoFrameworks())
import PtyLab
import PtyLab.Engines.GradientEngine
from PtyLab.Monitor.TensorboardMonitor import TensorboardMonitor, center_angle
assert "tensorflow" not in sys.modules
assert "torch" not in sys.modules

try:
    TensorboardMonitor()
except ImportError as exc:
    assert "pip install tensorflow" in str(exc), str(exc)
else:
    raise AssertionError("Expected TensorFlow installation guidance")

try:
    from PtyLab.Engines.GradientEngine import GradientEngine
except ImportError as exc:
    assert "pip install torch" in str(exc), str(exc)
else:
    raise AssertionError("Expected PyTorch installation guidance")
'''
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
