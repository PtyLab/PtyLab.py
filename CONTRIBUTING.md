## Package management with Poetry

The build-system as given under [`pyproject.toml`](pyproject.toml) is based on [Poetry](https://python-poetry.org/), a Python package manager. If you are a would like to modify existing dependencies or add new ones, relying on Poetry is recommended. It comes with its own dependency resolver, making sure nothing breaks. We recommend using `conda` as an environment manager and `poetry` as a dependency resolver.

First clone this repository and create a conda environment as explained in the development section of `README.md`. Install `poetry` from this [installation guide](https://python-poetry.org/docs/#installing-with-pipx). 

At the root of the repository, activate the conda environment and install `ptylab` and its depedencies with `poetry`.

```bash
conda activate ptylab_venv
poetry install -E dev
```

This will also create a `poetry.lock` file that contains the list of all the *pinned dependencies* as given under `pyproject.toml`.

If you want to install a new package from [PyPI](https://pypi.org/project/pip/), please do so as

```bash
poetry add <package-name>
``` 

This will not just install the new package, but also resolve the existing environment and make sure no other dependencies break. Similarly, you can remove a package as `poetry remove <package-name>`. For more information, please rely on their [documentation](https://python-poetry.org/docs/basic-usage/). 