# Contributing

This page is for contributors working on PtyLab.py itself. To use PtyLab, see [Installation](getting-started/installation.md).

Contributing follows three steps: [set up](#1-set-up) a local copy, [make your changes](#2-make-changes) with tests, and [submit a pull request](#3-submit-a-pull-request).

## 1. Set up

### Fork and clone

[Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) [PtyLab/PtyLab.py](https://github.com/PtyLab/PtyLab.py) on GitHub, then clone your fork and add the original repository as `upstream`:

```bash
git clone git@github.com:<your-username>/PtyLab.py.git
cd PtyLab.py
git remote add upstream https://github.com/PtyLab/PtyLab.py.git
```

Use `https://github.com/<your-username>/PtyLab.py.git` instead if you have not set up SSH keys for GitHub.

Work on a new branch created from the latest `main`:

```bash
git fetch upstream
git switch -c my-feature upstream/main
```

### Install

Install the project with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` always includes the development tools (pytest, ipykernel, ruff); add the extras for your machine:

```bash
uv sync --all-extras                  # GPU machine: gpu + gui + torch
uv sync --all-extras --no-extra gpu   # CPU-only machine
```

This creates a `.venv` virtual environment in the project root. Select it in your IDE or activate it:

```bash
source .venv/bin/activate
```

`uv sync` removes any extra you do not list, so repeat the same flags whenever you sync again.

### PyTorch build

The `torch` extra is needed for `GradientEngine`. Unlike `uv pip install`, `uv sync` takes no `--torch-backend` flag: on Linux and Windows it always installs the PyTorch build for CUDA 12.6, set in `[tool.uv.sources]` of `pyproject.toml`. On macOS it installs the standard build from PyPI, and `GradientEngine` runs on the CPU.

The CUDA build also runs on a CPU-only machine, falling back to the CPU. To skip the large download, add `--no-extra torch`; the `GradientEngine` tests are then skipped.

Check which GPUs CuPy and PyTorch detect with:

```bash
uv run ptylab check gpu
```

To target a different CUDA version, change the `pytorch-cu126` index URL in `pyproject.toml`. Keep it below CUDA 13.0, matching the CUDA 12 CuPy build of the `gpu` extra.

## 2. Make changes

Avoid breaking changes to the public API. If one is unavoidable, keep the old name working and emit a `DeprecationWarning` that names the replacement (see `pcPIE` in `PtyLab/Engines/mPIE.py`).

### Code style

Format the files you changed with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff format path/to/changed_file.py
```

### Tests

Add tests for new implementations and run the suite with:

```bash
uv run pytest tests
```

Tests that need an unavailable dependency are skipped, for example the `GradientEngine` tests without the `torch` extra and the CUDA tests on a machine without a GPU. Check the skipped count in the summary to see what was not exercised.

### Documentation

Documentation changes are deployed automatically once a pull request is merged into `main`. Check them locally first:

```bash
uv run --group docs mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The page reloads when you save a change.

## 3. Submit a pull request

### Using LLMs

You may use LLMs to generate code, as long as the pull request:

- is of reasonable size,
- contains well-written code,
- is logically self-contained, addressing one feature or fix,
- adds appropriate tests for new functions.

By opening a pull request, you confirm that you understand all of the code in it. Very large pull requests will be closed, with a request to resubmit them as smaller, logically separate pull requests.

### Opening the pull request

Push your branch to your fork and open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) against `main` of PtyLab/PtyLab.py:

```bash
git push -u origin my-feature
```

GitHub fills the description with the [pull request template](https://github.com/PtyLab/PtyLab.py/blob/main/.github/pull_request_template.md). Fill in each section rather than replacing it: why the change is needed, how it was tested (a demo figure or screenshot helps), and any notes such as deprecations or changed reconstruction results. Keep the description short: list the changes per file as brief bullet points in simple words.

If `main` has moved on in the meantime, update your branch with `git fetch upstream` and `git rebase upstream/main`.
