# Installation

PtyLab.py requires **Python 3.10–3.13** and is distributed on [PyPI](https://pypi.org/project/ptylab/).

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

## Optional extras

The core install is deliberately small. Features that need heavier packages are opt-in extras:

| Extra | Installs | Needed for |
|---|---|---|
| `gpu` | CuPy (CUDA 12) | GPU reconstruction |
| `gui` | pyqtgraph, PySide6 | the interactive Qt viewer (`show3Dslider` outside a notebook) |

Combine them in one command, for example:

```bash
uv pip install "ptylab[gpu,gui]"
```

## Interactive viewer (Qt)

Outside a notebook, `show3Dslider` opens an interactive pyqtgraph (Qt) viewer when the `gui` extra is installed. Without it, or if Qt cannot start, it falls back to a matplotlib slider.

On a remote machine, the viewer window needs a graphical connection: a remote desktop session (or equivalent), or X11 forwarding from Linux or macOS (`ssh -X`). Without a display, the matplotlib fallback is used.

## Latest development version

To install the unreleased state of `main` directly from GitHub:

```bash
uv pip install git+https://github.com/PtyLab/PtyLab.py.git
```

## Verify GPU detection

After installing with a CUDA extra, confirm the GPU is detected:

```bash
ptylab check gpu
```

This prints available GPU device information or warns if no GPU is found.

## Development setup

Clone the repository and install all optional dependencies (GPU, plotting, FPM calibration, docs, tests):

```bash
git clone https://github.com/PtyLab/PtyLab.py.git
cd PtyLab.py
uv sync --all-extras
```

`uv sync` makes the environment match exactly the extras you ask for, so install them all in one command. Running `uv sync --extra gui` afterwards would remove the others.

This creates a `.venv` virtual environment in the project root. Select it in your IDE or activate it:

```bash
source .venv/bin/activate
```

On a machine without a GPU, `uv sync --all-extras` still works: the CUDA packages are installed but unused. To keep the environment smaller, list only the extras you need, for example `uv sync --extra dev,gui`.

## Running tests

```bash
uv run pytest tests
```

## Serving documentation locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
