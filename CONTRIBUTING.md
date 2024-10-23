# Package Management with Poetry

If you intend to modify existing dependencies or add new ones, please **read this document carefully!**

Our build system, as specified in [`pyproject.toml`](pyproject.toml), is based on [Poetry](https://python-poetry.org/docs/), a Python package manager that includes its own dependency resolver to ensure compatibility among packages. We use `poetry` as a dependency resolver and `conda` as our environment manager.

Installing and familiarizing yourself with Poetry is recommended. The easiest way to install Poetry is via `pipx`. You can install `pipx` and Poetry with an export plugin as follows:

```bash
pip install pipx
pipx install poetry
pipx inject poetry poetry-plugin-export
```

Let's create the conda environment, assuming you have cloned the package and have navigated to the root of the repository,

```bash
conda create --name ptylab_venv python=3.11.5 # or python version satisfying ">=3.9, <3.12"
conda activate ptylab_venv
```

Within the environmemt, `ptylab` and its dependencies are installed using `poetry` along with the pre-commit hook,

```bash
poetry install --all-extras --sync
pre-commit install
```

This will utilize `pyproject.toml` and `poetry.lock` file for installing  *pinned dependencies*. 

For installing `cupy` for GPU usage, you can specify the optional `gpu-cuda11x` or `gpu-cuda12x` flag based on your [CUDA toolkit version](https://docs.cupy.dev/en/stable/install.html):

```bash
poetry install --all-extras --with gpu-cuda11x --sync
```

If you encounter installation issues, check your CUDA version with:

```bash
nvcc --version
```

If this command is not recognized, it's likely that your CUDA `PATH` and `LD_LIBRARY_PATH` are not set properly. For more information, refer to the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

## Modifying Packages

To install a new package from [PyPI](https://pypi.org/project/pip/), use:

```bash
poetry add <package-name>
```

This command will install the new package and resolve existing dependencies to prevent conflicts. Similarly, you can remove a package with:

```bash
poetry remove <package-name>
```

For more information, refer to the [Poetry documentation](https://python-poetry.org/docs/basic-usage/). These commands will modify the `pyproject.toml` and `poetry.lock` files. Ensure that you increment the package version (at least a minor version change) when making such modifications.

> [!WARNING] 
> When a dependency is changed, the `poetry.lock` file updates. If you try to commit these changes, the pre-commit hook might prevent it, especially if the `requirements.txt` file is modified. In this case, you need to stage both the `requirements.txt` and `poetry.lock` files for the commit to proceed.

If, for any reason, the pre-commit hook does not work, you can manually generate the `requirements.txt` file with:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes --extras dev
```

The `requirements.txt` file is essential for users who prefer not to use Poetry.