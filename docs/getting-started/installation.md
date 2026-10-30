# Installation

PtyLab.py requires **Python 3.11 or newer** and is distributed on [PyPI](https://pypi.org/project/ptylab/). Install it within a virtual environment of your liking (Ex: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [uv venv](https://docs.astral.sh/uv/pip/environments/)).

If you are contributing to the library instead, see the [contributing guide](../contributing.md).

## From PyPI

!!! tip
    The commands below use [uv](https://docs.astral.sh/uv/getting-started/installation/) for much faster installs. With plain `pip`, drop the `uv` prefix.

=== "CPU"

    ```bash
    uv pip install -U ptylab
    ```

=== "GPU"

    ```bash
    uv pip install -U "ptylab[gpu]" # installs cupy for CUDA 12
    ```

## Optional extras

These extras install optional features; each links to its section below:

| Extra | Installs | Needed for |
|---|---|---|
| `gpu` | CuPy (CUDA 12) | [GPU reconstruction](#verify-gpu-detection) |
| `torch` | PyTorch | [differentiable ptychography](#differentiable-ptychography) (`GradientEngine`) |
| `gui` | pyqtgraph, PySide6 | the [interactive Qt viewer](#interactive-viewer-qt) (`show3Dslider` outside a notebook) |

## Interactive viewer (Qt)

Outside a notebook, `show3Dslider` opens an interactive pyqtgraph (Qt) viewer when the `gui` extra is installed. Without it, or if Qt cannot start, it falls back to a matplotlib slider.

On a remote machine, the viewer window needs a graphical connection: a remote desktop session (or equivalent), or X11 forwarding from Linux or macOS (`ssh -X`). Without a display, the matplotlib fallback is used.

## Differentiable ptychography

The module `PtyLab.Engines.GradientEngine`, based on PyTorch, implements differentiable (gradient-based) ptychography so that custom models, losses and constraints can be used. It needs PyTorch, installed with the `torch` extra:

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

`--torch-backend=cu126` installs the CUDA 12 build of PyTorch, which matches CuPy and also runs on the CPU.

!!! warning
    `GradientEngine` is a work in progress. Its API should stay fixed, but this is not guaranteed for the time being. See [Differentiable Ptychography](../differentiable/overview.md) for the API.

## Latest unreleased version

To install the latest unreleased changes on `main` directly from GitHub:

```bash
uv pip install -U git+https://github.com/PtyLab/PtyLab.py.git
```

Extras work the same way, for example with GPU support:

```bash
uv pip install -U "ptylab[gpu] @ git+https://github.com/PtyLab/PtyLab.py.git"
```

## Verify GPU detection

After installing with the `gpu` extra, check whether the GPU is correctly configured in your environment with:

```bash
uv run ptylab check gpu
```

This prints the GPUs that CuPy and PyTorch (if installed) find and the CUDA version each uses. It fails if a library cannot use the GPU.

When CuPy finds a GPU, PtyLab uses it by default (`params.gpuSwitch = True`); set `params.gpuSwitch = False` to run on the CPU. The same code runs on both. For large datasets, `engine.convert2single()` switches to single precision (float32), which halves memory use and is faster on the GPU; call it after creating the engine and before `reconstruct()`.
