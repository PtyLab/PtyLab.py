# Development

This page is for contributors working on PtyLab.py itself. To use PtyLab, see [Installation](getting-started/installation.md).

## Setup

Clone the repository and install it with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` always includes the development tools (pytest, ipykernel); add the extras for your machine:

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
uv sync --all-extras                  # GPU machine: gpu + gui
uv sync --all-extras --no-extra gpu   # CPU-only machine
```

This creates a `.venv` virtual environment in the project root. Select it in your IDE or activate it:

```bash
source .venv/bin/activate
```

## Running tests

Add tests for new implementations and run the suite with:

```bash
uv run pytest tests
```

Tests that need an unavailable dependency are skipped, for example the CUDA tests on a machine without a GPU. Check the skipped count in the summary to see what was not exercised.

## Serving documentation locally

Documentation changes are deployed automatically once a pull request is merged into `main`. Check them locally first:

```bash
uv run --group docs mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The page reloads when you save a change.
