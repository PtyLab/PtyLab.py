# Contributing

This page is for contributors. To use PtyLab, see [Installation](getting-started/installation.md).

There are three steps: [set up](#1-set-up) a local copy, [make your changes](#2-make-changes) with tests, and [submit a pull request](#3-submit-a-pull-request).

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

Install the project with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` creates a virtual environment with the development tools (pytest, ruff), the Qt viewer and PyTorch; add the `gpu` extra on a machine with an NVIDIA GPU:

```bash
uv sync --extra gpu
```

PyTorch comes as the CUDA 12 build on Linux and Windows, which matches CuPy, and the PyPI build on macOS.

This creates a `.venv` in the project root; select it in your IDE or run `source .venv/bin/activate`. `uv sync` removes any extra you leave out, so reuse the same flags on every sync. Check which GPUs CuPy and PyTorch detect with:

```bash
uv run ptylab check gpu
```

## 2. Make changes

Avoid breaking changes to the public API. If one is unavoidable, keep the old name working and emit a `DeprecationWarning` that names the replacement (see `pcPIE` in `PtyLab/Engines/mPIE.py`).

### Code style

Format the files you changed with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff format path/to/changed_file.py
```

### Tests

Add tests for new code and run the suite with:

```bash
uv run pytest tests
```

Tests that need an unavailable dependency are skipped, for example the CUDA tests on a machine without a GPU. Check the skipped count in the summary to see what did not run.

### Documentation

The documentation is written in Markdown in `docs/`, and its navigation is set in `mkdocs.yml`. Changes go live when a pull request is merged into `main`. Preview them locally with:

```bash
uv run ptylab examples render          # creates the example pages, see Examples below
uv run --group docs zensical serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The page reloads when you save a change.

Write maths in LaTeX between `$...$` (inline) or `$$...$$` (display), and draw diagrams in a ` ```mermaid ` code block.

#### Examples

Each page under Examples is made from a Python script `docs/examples/scripts/example_<name>.py`, split into cells by `# %%` lines. The page shows the script with the outputs of its code cells (printed text and figures), which are saved in `docs/examples/outputs/example_<name>/`.

- After changing an example, run the same command as for the preview:
```bash
uv run ptylab examples render
```
It reruns only the examples whose script you changed, which can take a few minutes, and then updates the pages. Commit the script together with its folder in `docs/examples/outputs/`.


- If anything changes at the package level, re-render the affected example:
```bash
uv run ptylab examples render example_<name>
```

- To add an example, write the script, list it in `docs/examples/index.md` and in the `nav` of `mkdocs.yml`, and run `render`. If you add a feature such as an engine or a constraint, please add an example for it.

- Before you push, run the below command:
```bash
uv run ptylab examples check
```
It lists any example that still needs `render`, or that is missing from the index or the nav.

## 3. Submit a pull request

### Opening the pull request

Push your branch to your fork and open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) against `main` of PtyLab/PtyLab.py:

```bash
git push -u origin my-feature
```

Fill in each section of the [pull request template](https://github.com/PtyLab/PtyLab.py/blob/main/.github/pull_request_template.md): why the change is needed, how it was tested, and any notes. Keep it short and in simple words.

If `main` has moved on, update your branch with `git fetch upstream` and `git rebase upstream/main`.

### Using LLMs

You may use LLMs to generate code, as long as the pull request:

- is of reasonable size,
- contains well-written code,
- is logically self-contained, addressing one feature or fix,
- adds appropriate tests for new functions.

By opening a pull request, you confirm that you understand all of the code in it. Very large pull requests will be closed, with a request to resubmit them as smaller, logically separate pull requests.
