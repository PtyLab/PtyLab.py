# Installation

PtyLab.py requires **Python 3.11 or newer** and is distributed on [PyPI](https://pypi.org/project/ptylab/). Install it within a virtual environment of your liking (Ex: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [uv venv](https://docs.astral.sh/uv/pip/environments/)).

If you are contributing to the library instead, see the [contributing guide](../contributing.md).

## From PyPI

!!! tip
    The commands below use [uv](https://docs.astral.sh/uv/getting-started/installation/) for much faster installs. If you don't have it, drop the `uv` prefix and use plain `pip`.

=== "CPU"

    ```bash
    uv pip install -U ptylab
    ```

=== "GPU"

    ```bash
    uv pip install -U "ptylab[cupy]"
    ```

The `cupy` extra installs [CuPy](https://cupy.dev/) for CUDA 12 and enables GPU acceleration of the reconstruction engines. CUDA 13 is not supported yet. The extra is not installed on macOS, where PtyLab runs on the CPU.

## Optional extras

These extras install optional features:

| Extra | Installs | Needed for |
|---|---|---|
| `cupy` | CuPy (CUDA 12) | GPU reconstruction |
| `gui` | pyqtgraph, PySide6 | the interactive Qt viewer (`show3Dslider` outside a notebook) |
| `torch` | PyTorch | differentiable ptychography (`GradientEngine`) |

For example to combine them in one command:

```bash
uv pip install -U "ptylab[cupy,gui]"
```

## Interactive viewer (Qt)

Outside a notebook, `show3Dslider` opens an interactive pyqtgraph (Qt) viewer when the `gui` extra is installed. Without it, or if Qt cannot start, it falls back to a matplotlib slider.

On a remote machine, the viewer window needs a graphical connection: a remote desktop session (or equivalent), or X11 forwarding from Linux or macOS (`ssh -X`). Without a display, the matplotlib fallback is used.

## Differentiable ptychography

The module `PtyLab.Engines.GradientEngine`, based on PyTorch, implements differentiable (gradient-based) ptychography so that custom models, losses and constraints can be used. Install it with the `torch` extra:

```bash
uv pip install -U "ptylab[torch]" --torch-backend=cu126   # PyTorch on CUDA 12.6
uv pip install -U "ptylab[torch]" --torch-backend=cpu     # no GPU
```

`--torch-backend=cu126` selects the PyTorch build for CUDA 12.6. Newer PyTorch builds also work, as long as they target a CUDA version below 13.0. Add the `cupy` extra (`ptylab[cupy,torch]`) to also run the classic engines on the GPU.

`GradientEngine` runs on the GPU through PyTorch whenever CUDA is available, independently of CuPy, and falls back to the CPU otherwise.

!!! warning
    `GradientEngine` is a work in progress. Its API should stay fixed, but this is not guaranteed for the time being. See [Differentiable Ptychography](../differentiable/overview.md) for the API.

## Latest unreleased version

To install the latest unreleased changes on `main` directly from GitHub:

```bash
uv pip install -U git+https://github.com/PtyLab/PtyLab.py.git
```

Extras work the same way, for example with GPU support:

```bash
uv pip install -U "ptylab[cupy] @ git+https://github.com/PtyLab/PtyLab.py.git"
```

## Verify GPU detection

After installing with the `cupy` extra, check whether the GPU is correctly configured:

```bash
uv run ptylab check gpu
```

This prints the CuPy CUDA device information, or warns if no GPU is found.

When CuPy finds a GPU, PtyLab uses it by default (`params.gpuSwitch = True`); set `params.gpuSwitch = False` to run on the CPU. The same code runs on both. For large datasets, `engine.convert2single()` switches to single precision (float32), which halves memory use and is faster on the GPU; call it after creating the engine and before `reconstruct()`.
