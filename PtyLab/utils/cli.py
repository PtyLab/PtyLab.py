import argparse
import sys


def _check_cupy_gpu():
    try:
        import cupy as cp

        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            print("cupy is installed but no CUDA devices found.")
            return False
        print(f"GPU available: {n} device(s)")
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode()
            mem_gb = props["totalGlobalMem"] / 1024**3
            print(f"  [{i}] {name}  ({mem_gb:.1f} GB)")

        if cp.is_available():
            print("cupy GPU: OK")
            return True
        else:
            print("cupy is installed but GPU is not available.")
            return False
    except ImportError:
        print(
            "cupy is not installed. If you are using a GPU, please check"
            " the readme for the appropriate cupy flag for your CUDA version."
        )
        return False
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False


def _check_torch_gpu():
    try:
        import torch
    except ModuleNotFoundError as e:
        if e.name == "torch":
            print("PyTorch is not installed; skipping PyTorch GPU check.")
            return True
        print(f"PyTorch GPU check failed: {e}")
        return False
    except Exception as e:
        print(f"PyTorch GPU check failed: {e}")
        return False

    try:
        if not torch.cuda.is_available():
            print("PyTorch is installed but CUDA GPU is not available.")
            return False
        n = torch.cuda.device_count()
        print(f"PyTorch CUDA GPU available: {n} device(s)")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  [{i}] {props.name}  ({mem_gb:.1f} GB)")
        print("PyTorch CUDA GPU: OK")
        return True
    except Exception as e:
        print(f"PyTorch GPU check failed: {e}")
        return False


def cmd_check_gpu():
    cupy_ok = _check_cupy_gpu()
    torch_ok = _check_torch_gpu()
    if not (cupy_ok and torch_ok):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="ptylab")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="System checks")
    check_sub = check.add_subparsers(dest="target")
    check_sub.add_parser(
        "gpu", help="Check CuPy and optional PyTorch CUDA GPU availability"
    )

    args = parser.parse_args()

    if args.command == "check" and args.target == "gpu":
        cmd_check_gpu()
    else:
        parser.print_help()
        sys.exit(1)
