import argparse
import sys


def cmd_check_gpu():
    try:
        import cupy as cp

        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            print("cupy is installed but no CUDA devices found.")
            sys.exit(1)
        print(f"GPU available: {n} device(s)")
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode()
            mem_gb = props["totalGlobalMem"] / 1024**3
            print(f"  [{i}] {name}  ({mem_gb:.1f} GB)")

        if cp.is_available():
            print("cupy GPU: OK")
        else:
            print("cupy is installed but GPU is not available.")
            sys.exit(1)
    except ImportError:
        print(
            "cupy is not installed. If you are using a GPU, please check"
            " the readme for the appropriate cupy flag for your CUDA version."
        )
        sys.exit(1)
    except Exception as e:
        print(f"GPU check failed: {e}")
        sys.exit(1)


def cmd_docs(args):
    # jupytext and nbclient come with the dev group, which a plain `pip install ptylab` lacks
    try:
        from PtyLab.utils import docs
    except ImportError as e:
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
    check_sub.add_parser("gpu", help="Check GPU / cupy availability")

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
