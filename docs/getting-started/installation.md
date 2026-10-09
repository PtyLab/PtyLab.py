# Installation

PtyLab.py requires **Python 3.10–3.13** and is distributed on [PyPI](https://pypi.org/project/ptylab/). Install it within a virtual environment.

## From PyPI

!!! tip
    The commands below use [uv](https://docs.astral.sh/uv/getting-started/installation/) for much faster installs. If you don't have it, drop the `uv` prefix and use plain `pip`.

=== "CPU"

    ```bash
    uv pip install ptylab
    ```

=== "GPU"

    ```bash
    uv pip install "ptylab[gpu]"
    ```

The `gpu` extra installs [CuPy](https://cupy.dev/) for CUDA 12 and enables GPU acceleration of the reconstruction engines. CUDA 13 is not supported yet. The extra is not installed on macOS, where PtyLab runs on the CPU.

## Optional extras

Some features can be installed with the optional dependencies with the following extras:

| Extra | Installs | Needed for |
|---|---|---|
| `gpu` | CuPy (CUDA 12) | GPU reconstruction |
| `gui` | pyqtgraph, PySide6 | the interactive Qt viewer (`show3Dslider` outside a notebook) |

For example to combine them in one command:

```bash
uv pip install "ptylab[gpu,gui]"
```

## Interactive viewer (Qt)

Outside a notebook, `show3Dslider` opens an interactive pyqtgraph (Qt) viewer when the `gui` extra is installed. Without it, or if Qt cannot start, it falls back to a matplotlib slider.

On a remote machine, the viewer window needs a graphical connection: a remote desktop session (or equivalent), or X11 forwarding from Linux or macOS (`ssh -X`). Without a display, the matplotlib fallback is used.

## Latest unreleased version

To install the latest unreleased changes on `main` directly from GitHub:

```bash
uv pip install git+https://github.com/PtyLab/PtyLab.py.git
```

Extras work the same way, for example with GPU support:

```bash
uv pip install "ptylab[gpu] @ git+https://github.com/PtyLab/PtyLab.py.git"
```

## Verify GPU detection

After installing with the `gpu` extra, check whether the GPU is correctly configured:

```bash
uv run ptylab check gpu
```

This prints the CuPy CUDA device information, or warns if no GPU is found.
