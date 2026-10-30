# Contributing

This page is for contributors of this library. To use PtyLab, see [Installation](getting-started/installation.md).

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

Add tests for new implementations and run the suite with:

```bash
uv run pytest tests
```

Tests that need an unavailable dependency are skipped, for example the CUDA tests on a machine without a GPU. Check the skipped count in the summary to see what did not run.

### Documentation

Documentation changes are deployed automatically once a pull request is merged into `main`. Check them locally first:

```bash
uv run --group docs zensical serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The page reloads when you save a change.

Write maths in LaTeX between `$...$` (inline) or `$$...$$` (display), and draw diagrams in a ` ```mermaid ` code block.

### Tutorials

Each tutorial is a Python script `tutorial_<name>.py` in `docs/tutorials/scripts/`, split into cells by `# %%` lines. If you add or change any of these scripts, re-render the tutorials with:

```bash
uv run ptylab docs render              # every tutorial whose script is new or changed
```

For rendering just a specific one, even if its script did not change (for example, after changing the code it uses):

```bash
uv run ptylab docs render <tutorial-name> # single tutorial (with or without .py)
```

This runs the script and saves the page and its figures in `docs/tutorials/` (`<tutorial-name>.md` and `<tutorial-name>_files/`), which should be committed as well. The `.ipynb` it also writes there is not committed. In the end, run `uv run ptylab docs check`, which reports pages that are out of date and tutorials missing from the index or the nav, as CI does.

New feature, such as an engine or a constraint? Please add a tutorial for it. List it in `docs/tutorials/index.md` and in the `nav` of `mkdocs.yml`.

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

GitHub fills the description with the [pull request template](https://github.com/PtyLab/PtyLab.py/blob/main/.github/pull_request_template.md). Fill in each section rather than replacing it: why the change is needed, how it was tested (a demo figure or screenshot helps), and any notes such as deprecations or changed reconstruction results. Keep the description short: list the changes per file as brief bullet points in simple words. Preferably write it yourself. If an LLM drafts it, edit it until it is easy to read, without unnecessary length or jargon.

If `main` has moved on in the meantime, update your branch with `git fetch upstream` and `git rebase upstream/main`.
