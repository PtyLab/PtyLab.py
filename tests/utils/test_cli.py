import sys
from types import SimpleNamespace

import pytest

from PtyLab.utils.cli import cmd_check_gpu

# The supported setup: CUDA 13 driver, CuPy (CUDA 12.9) and PyTorch (CUDA 12.6)


def fake_cupy(cuda=12090):
    return SimpleNamespace(
        __version__="14.2.0",
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 1,
                getDeviceProperties=lambda _: {
                    "name": b"Test GPU",
                    "totalGlobalMem": 8 * 1024**3,
                },
                driverGetVersion=lambda: 13020,
                runtimeGetVersion=lambda: cuda,
            )
        ),
        is_available=lambda: True,
    )


def fake_torch(available=True, cuda="12.6"):
    return SimpleNamespace(
        __version__="2.14.1",
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(
                name="Test GPU", total_memory=8 * 1024**3
            ),
        ),
    )


def run_check(monkeypatch, capsys, cupy, torch):
    """Run `ptylab check gpu` with fake modules (None = not installed); return (exit code, output)."""
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "torch", torch)
    try:
        cmd_check_gpu()
        code = 0
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


@pytest.mark.parametrize(
    "cupy_available,torch_available,expected_exit",
    [
        (True, True, 0),
        (True, False, 1),
        (True, None, 0),  # PyTorch is optional
        (False, True, 1),
        (False, False, 1),
        (False, None, 1),
    ],
)
def test_gpu_checks_report_both_backends(
    monkeypatch, capsys, cupy_available, torch_available, expected_exit
):
    cupy = fake_cupy() if cupy_available else None
    torch = fake_torch(torch_available) if torch_available is not None else None
    code, output = run_check(monkeypatch, capsys, cupy, torch)
    assert code == expected_exit
    assert ("cupy GPU: OK" if cupy_available else "cupy is not installed") in output
    if cupy_available:
        assert "NVIDIA driver supports CUDA 13.2" in output
        assert "cupy 14.2.0 uses CUDA 12.9" in output
    if torch_available is None:
        assert "PyTorch is not installed" in output
    elif torch_available:
        assert "PyTorch 2.14.1 uses CUDA 12.6" in output
        assert "PyTorch GPU: OK" in output
        assert "[0] Test GPU  (8.0 GB)" in output
    else:
        assert "PyTorch cannot use the GPU" in output
    assert "Note:" not in output  # matching CUDA majors


def test_torch_runtime_error_is_reported(monkeypatch, capsys):
    def unavailable():
        raise RuntimeError("driver initialization failed")

    torch = fake_torch()
    torch.cuda.is_available = unavailable
    code, output = run_check(monkeypatch, capsys, None, torch)
    assert code == 1
    assert "PyTorch GPU check failed: driver initialization failed" in output


def test_cuda_major_mismatch_is_a_note_when_both_work(monkeypatch, capsys):
    # e.g. plain pip: CUDA 13 PyTorch from PyPI on a CUDA 13 driver
    code, output = run_check(monkeypatch, capsys, fake_cupy(), fake_torch(cuda="13.2"))
    assert code == 0
    assert "Note: cupy uses CUDA 12 but PyTorch uses CUDA 13" in output
    assert "--torch-backend=cu126" in output


def test_cuda_major_mismatch_fails_when_torch_cannot_use_the_gpu(monkeypatch, capsys):
    # e.g. CUDA 13 PyTorch on a CUDA 12 driver
    torch = fake_torch(available=False, cuda="13.2")
    code, output = run_check(monkeypatch, capsys, fake_cupy(), torch)
    assert code == 1
    assert "--torch-backend=cu126" in output


def test_torch_cpu_build_is_reported(monkeypatch, capsys):
    code, output = run_check(monkeypatch, capsys, fake_cupy(), fake_torch(cuda=None))
    assert code == 1
    assert "PyTorch 2.14.1 is the CPU build" in output
