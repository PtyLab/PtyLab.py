import sys
from types import SimpleNamespace

import pytest

from PtyLab.utils.cli import cmd_check_gpu


@pytest.mark.parametrize(
    "cupy_available,torch_available,expected_exit",
    [
        (True, True, 0),
        (True, False, 1),
        (True, None, 0),
        (False, True, 1),
        (False, False, 1),
        (False, None, 1),
    ],
)
def test_gpu_checks_report_both_backends(
    monkeypatch, capsys, cupy_available, torch_available, expected_exit
):
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 1,
                getDeviceProperties=lambda i: {
                    "name": b"Test GPU",
                    "totalGlobalMem": 8 * 1024**3,
                },
            )
        ),
        is_available=lambda: True,
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: torch_available,
            device_count=lambda: 1,
            get_device_properties=lambda i: SimpleNamespace(
                name="Test GPU", total_memory=8 * 1024**3
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy if cupy_available else None)
    monkeypatch.setitem(
        sys.modules, "torch", torch if torch_available is not None else None
    )
    if expected_exit:
        with pytest.raises(SystemExit) as exc:
            cmd_check_gpu()
        assert exc.value.code == expected_exit
    else:
        cmd_check_gpu()
    output = capsys.readouterr().out
    assert ("cupy GPU: OK" if cupy_available else "cupy is not installed") in output
    if torch_available is None:
        assert "PyTorch is not installed; skipping" in output
    elif torch_available:
        assert "PyTorch CUDA GPU: OK" in output
        assert "[0] Test GPU  (8.0 GB)" in output
    else:
        assert "PyTorch is installed but CUDA GPU is not available" in output


def test_torch_runtime_error_is_reported_after_cupy_failure(monkeypatch, capsys):
    def unavailable():
        raise RuntimeError("driver initialization failed")

    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=unavailable)),
    )
    with pytest.raises(SystemExit) as exc:
        cmd_check_gpu()
    assert exc.value.code == 1
    assert (
        "PyTorch GPU check failed: driver initialization failed"
        in capsys.readouterr().out
    )
