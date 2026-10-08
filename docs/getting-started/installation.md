# Installation

PtyLab.py requires **Python 3.10–3.13** and is distributed on [PyPI](https://pypi.org/project/ptylab/). Install it within a virtual environment.

## From PyPI (pip)

=== "CPU"

    ```bash
    pip install ptylab
    ```

=== "GPU"

    ```bash
    pip install "ptylab[gpu]"
    ```

!!! tip
    For faster installs, we recommend [uv](https://docs.astral.sh/uv/getting-started/installation/):

    ```bash
    uv pip install ptylab
    ```

The `gpu` extra installs [CuPy](https://cupy.dev/) for CUDA 12 and enables GPU acceleration of the reconstruction engines. CUDA 13 is not supported yet. The extra is not installed on macOS, where PtyLab runs on the CPU.

## Latest unreleased version

To install the latest unreleased changes on `main` directly from GitHub:

```bash
uv pip install git+https://github.com/PtyLab/PtyLab.py.git
```

Extras work the same way, for example with GPU support:

```bash
uv pip install "ptylab[gpu] @ git+https://github.com/PtyLab/PtyLab.py.git"
```

## Optional: differentiable ptychography

The module `PtyLab.Engines.GradientEngine`, based on PyTorch, implements differentiable (gradient-based) ptychography so that custom models, losses and constraints can be used. Install it with the `torch` extra:

```bash
uv pip install "ptylab[gpu,torch]" --torch-backend=cu126
```

`--torch-backend=cu126` selects the PyTorch build for CUDA 12.6. Newer PyTorch builds also work, as long as they target a CUDA version below 13.0. Omit `gpu` and use `--torch-backend=cpu` on a machine without a GPU.

`GradientEngine` runs on the GPU through PyTorch whenever CUDA is available, independently of CuPy, and falls back to the CPU otherwise.

!!! warning
    `GradientEngine` is a work in progress. Its API should stay fixed, but this is not guaranteed for the time being. See the [GradientEngine usage guide](https://github.com/PtyLab/PtyLab.py/blob/main/PtyLab/Engines/GradientEngine/README.md) for the API.

## Verify GPU detection

After installing with the `gpu` or `torch` extra, check whether the GPU is correctly configured:

```bash
uv run ptylab check gpu
```

This prints the CuPy and, if installed, PyTorch CUDA device information, or warns if no GPU is found.

## Development setup

Clone the repository and install the development and GPU dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
uv sync --extra dev --extra gpu # omit --extra gpu if you are on CPU
```

Use `https://github.com/PtyLab/PtyLab.py.git` instead if you have not set up SSH keys for GitHub.

This creates a `.venv` virtual environment in the project root. Select it in your IDE or activate it:

```bash
source .venv/bin/activate
```

Add `--extra torch` for `GradientEngine` development:

```bash
uv sync --extra dev --extra gpu --extra torch
```

`uv sync` installs exactly the extras you list, so repeat all of them whenever you sync again.

## Running tests

Add tests for new implementations and run the suite with:

```bash
uv run pytest tests
```

Tests that need an unavailable dependency are skipped rather than failed: the `GradientEngine` tests without the `torch` extra, and the CUDA tests on a machine without a GPU. Check the skipped count in the summary to see what was not exercised.

## Serving documentation locally

Documentation changes are deployed automatically once a pull request is merged into `main`. Check them locally first:

```bash
uv run --extra docs mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The page reloads when you save a change.
