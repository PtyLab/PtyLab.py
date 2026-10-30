import argparse
import sys

# CuPy and PyTorch both as CUDA 12 builds (PyPI only has CUDA 13 builds of PyTorch)
GPU_INSTALL = 'uv pip install -U "ptylab[gpu,torch]" --torch-backend=cu126'


def _cuda_version(v):
    # CUDA encodes versions as 1000 * major + 10 * minor, e.g. 13020 -> "13.2"
    return f"{v // 1000}.{v % 1000 // 10}"


def _print_devices(lib, devices):
    """Print the `(name, memory in bytes)` of each GPU a library finds."""
    print(f"{lib} GPU available: {len(devices)} device(s)")
    for i, (name, mem) in enumerate(devices):
        print(f"  [{i}] {name}  ({mem / 1024**3:.1f} GB)")


def _check_cupy_gpu():
    """Print the CuPy GPU status; return (ok, CUDA major version of CuPy or None)."""
    try:
        import cupy as cp
    except ImportError:
        print(
            f"cupy is not installed. On an NVIDIA GPU, install it with `{GPU_INSTALL}`."
        )
        return False, None

    try:
        rt = cp.cuda.runtime
        driver, runtime = rt.driverGetVersion(), rt.runtimeGetVersion()
        print(f"NVIDIA driver supports CUDA {_cuda_version(driver)}")
        print(f"cupy {cp.__version__} uses CUDA {_cuda_version(runtime)}")
        props = [rt.getDeviceProperties(i) for i in range(rt.getDeviceCount())]
        _print_devices(
            "cupy", [(p["name"].decode(), p["totalGlobalMem"]) for p in props]
        )
        if not (props and cp.is_available()):
            print("cupy cannot use the GPU.")
            return False, None
        print("cupy GPU: OK")
        return True, runtime // 1000
    except Exception as e:  # noqa: BLE001 - report any CUDA or driver error, never crash
        print(f"cupy GPU check failed: {e}")
        return False, None


def _check_torch_gpu():
    """Print the PyTorch GPU status; return (ok, CUDA major version of PyTorch or None)."""
    try:
        import torch
    except ModuleNotFoundError as e:
        if e.name != "torch":
            print(f"PyTorch GPU check failed: {e}")
            return False, None
        # PyTorch is optional: only GradientEngine needs it
        print("PyTorch is not installed (optional, `torch` extra); skipping its check.")
        return True, None

    try:
        # torch.version.cuda is the CUDA version of the build, None for the CPU build
        if torch.version.cuda is None:
            print(
                f"PyTorch {torch.__version__} is the CPU build. Reinstall with `{GPU_INSTALL}`."
            )
            return False, None
        print(f"PyTorch {torch.__version__} uses CUDA {torch.version.cuda}")
        major = int(torch.version.cuda.split(".")[0])
        if not torch.cuda.is_available():
            print("PyTorch cannot use the GPU.")
            return False, major
        props = [
            torch.cuda.get_device_properties(i)
            for i in range(torch.cuda.device_count())
        ]
        _print_devices("PyTorch", [(p.name, p.total_memory) for p in props])
        print("PyTorch GPU: OK")
        return True, major
    except Exception as e:  # noqa: BLE001 - report any CUDA or driver error, never crash
        print(f"PyTorch GPU check failed: {e}")
        return False, None


def cmd_check_gpu():
    cupy_ok, cupy_major = _check_cupy_gpu()
    torch_ok, torch_major = _check_torch_gpu()
    # A mixed CUDA major is not an error by itself: a CUDA 13 driver runs both. It
    # happens with plain pip, where PyTorch comes from PyPI (CUDA 13 on Linux).
    if cupy_major and torch_major and cupy_major != torch_major:
        print(
            f"Note: cupy uses CUDA {cupy_major} but PyTorch uses CUDA {torch_major}."
            f" For matching CUDA 12 builds, reinstall with `{GPU_INSTALL}`."
        )
    if not (cupy_ok and torch_ok):
        sys.exit(1)


def cmd_docs(args):
    # jupytext and nbclient come with the dev group, which a plain `pip install ptylab` lacks
    try:
        from PtyLab.utils import docs
    except ImportError as e:
        if e.name not in ("jupytext", "nbclient", "nbformat"):
            raise  # a real error, e.g. a missing PtyLab module
        print(
            f"{e.name} is not installed. Run `uv sync` in a checkout of PtyLab.py first."
        )
        sys.exit(1)

    if args.target == "render":
        sys.exit(docs.render_command(args.names, args.all, args.timeout))
    if args.target == "notebooks":
        sys.exit(docs.notebooks_command())
    sys.exit(docs.check_command())


def main():
    parser = argparse.ArgumentParser(prog="ptylab")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="System checks")
    check_sub = check.add_subparsers(dest="target")
    check_sub.add_parser(
        "gpu", help="Check CuPy and PyTorch GPU availability and CUDA versions"
    )

    docs = sub.add_parser(
        "docs", help="Build the tutorial pages from docs/tutorials/scripts/"
    )
    docs_sub = docs.add_subparsers(dest="target")
    render = docs_sub.add_parser(
        "render",
        help="Run tutorial scripts into docs pages (default: those out of date)",
    )
    render.add_argument(
        "names", nargs="*", help="tutorials to render, e.g. tutorial_CPM"
    )
    render.add_argument(
        "--all", action="store_true", help="render every tutorial, even if up to date"
    )
    render.add_argument(
        "--timeout", type=int, default=3600, help="seconds allowed per cell"
    )
    docs_sub.add_parser(
        "notebooks",
        help="Convert the tutorial scripts into download notebooks (not executed)",
    )
    docs_sub.add_parser(
        "check", help="Check that every page matches its script and is in the nav"
    )

    args = parser.parse_args()

    if args.command == "check" and args.target == "gpu":
        cmd_check_gpu()
    elif args.command == "docs" and args.target in ("render", "notebooks", "check"):
        cmd_docs(args)
    else:
        parser.print_help()
        sys.exit(1)
