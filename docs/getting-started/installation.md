# Installation

PtyLab.py requires **Python 3.11 or newer** and is distributed on [PyPI](https://pypi.org/project/ptylab/). Install it in a virtual environment, for example with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [uv venv](https://docs.astral.sh/uv/pip/environments/).

If you are contributing to the library instead, see the [contributing guide](../contributing.md).

## From PyPI

!!! tip
    The commands below use [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs faster. With plain `pip`, drop the `uv` prefix.

=== "CPU"

    ```bash
    uv pip install -U ptylab
    ```

=== "GPU"

    ```bash
    uv pip install -U "ptylab[gpu]" # installs cupy for CUDA 12
    ```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `gpu` | CuPy (CUDA 12) | GPU reconstruction |
| `torch` | PyTorch | differentiable ptychography (`GradientEngine`) |
| `gui` | pyqtgraph, PySide6 | the interactive Qt viewer |

## Verify GPU detection

After installing with the `gpu` extra, run:

```bash
uv run ptylab check gpu
```

It lists the GPUs that CuPy and PyTorch (if installed) find, with their CUDA versions, and fails if a library cannot use the GPU.

PtyLab uses the GPU when CuPy finds one; set `params.gpuSwitch = False` to run on the CPU. For large datasets, call `engine.convert2single()` before `reconstruct()` to use single precision, which halves memory use.

## Differentiable ptychography

`GradientEngine` reconstructs with PyTorch gradients, so custom models, losses and constraints can be used. Install it with the `torch` extra:

=== "GPU (Linux, Windows)"

    ```bash
    uv pip install -U "ptylab[gpu,torch]" --torch-backend=cu126
    ```

=== "CPU (Linux, Windows)"

    ```bash
    uv pip install -U "ptylab[torch]" --torch-backend=cu126
    ```

=== "macOS"

    ```bash
    uv pip install -U "ptylab[torch]"
    ```

`--torch-backend=cu126` picks the CUDA 12 build of PyTorch, which matches CuPy and also runs on the CPU.

## Interactive viewer (Qt)

With the `gui` extra, `show3Dslider` opens a Qt viewer outside a notebook; otherwise it falls back to a matplotlib slider. On a remote machine the window needs a display, such as a remote desktop or `ssh -X`.

## Latest unreleased version

To install the latest unreleased changes on `main` directly from GitHub:

```bash
uv pip install -U git+https://github.com/PtyLab/PtyLab.py.git
```

Extras work the same way, for example with GPU support:

```bash
uv pip install -U "ptylab[gpu] @ git+https://github.com/PtyLab/PtyLab.py.git"
```
