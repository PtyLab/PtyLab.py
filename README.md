# PtyLab.py
![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)

PtyLab is an inverse modeling toolbox for Conventional (CP) and Fourier (FP) ptychography in a unified framework. For more information please check the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).
 
## Installation

To install the package from source,

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git
```

This package uses `cupy` to utilize GPU for faster reconstruction. Please check their [instructions](https://docs.cupy.dev/en/stable/install.html) for installing this dependency.

### Development
 
Please clone this repository and navigate to the root folder
```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
```

Inside a virtual environment (preferably with conda), please install `ptylab` and its dependencies:
```bash
conda create --name ptylab_venv python=3.11.5 # or python version satisfying ">=3.9, <3.12"
conda activate ptylab_venv
pip install -e .[dev]
```

To use the GPU, `cupy` can be additionally installed in this environment.

## Getting started

The simplest way to get started is by simulating some ptychography data. Assuming you are at the root directory and within the environment, please execute the following command.

```bash
python example_scripts/simulateData.py
```
This will store the data under the [example_data](example_data) directory as `simu.hdf5`. To reconstruct this data and store as `recon.hdf5` in the same directory,

```bash
python example_scripts/exampleReconstructionCPM.py --f example_data/simu.hdf5
```
To use GPU, the above command can be appended with `--gpu` flag.

Under [example_scripts](example_scripts/) and [jupyter_tutorials](jupyter_tutorials), you can find examples for typical use cases. 

## Package management with conda and poetry

> [!NOTE]
> The build-system as given under [`pyproject.toml`](pyproject.toml) is based on [Poetry](https://python-poetry.org/), a Python package manager. If you are a maintainer of `PtyLab.py` and would like to modify existing packages or add new ones, relying on Poetry for development is recommended. It comes with its own dependency resolver, making sure nothing breaks.

If there is no existing conda environment, please create one and install `poetry` within the environment.

```bash
conda create --name ptylab_venv python=3.11.5 # or python version satisfying ">=3.9, <3.12"
conda install poetry
```

Within the conda virtual environment, you can now install `PtyLab.py` and its depedencies with poetry,

```bash
conda activate ptylab_venv
poetry install
```

This will also create a `poetry.lock` file that contains the list of all the *pinned dependencies* as given under `pyproject.toml`. Sometimes Poetry fails to install a dependency as it tries to be compatible with all OS platforms. In this case, please install that failed dependency with `pip`.

If you want to install a new package from [PyPI](https://pypi.org/project/pip/), please do so with `poetry`.

```bash
poetry add <package-name>
``` 

This will not just install the new package, but also resolve the existing environment and make sure no other dependencies break. Similarly, you can remove a package as `poetry remove <package-name>`. For more information, please rely on the [Poetry](https://python-poetry.org/) documentation. 

## Citation

If you use this package in your work, cite us as below. 

```tex
@article{Loetgering:23,
        author = {Lars Loetgering and Mengqi Du and Dirk Boonzajer Flaes and Tomas Aidukas and Felix Wechsler and Daniel S. Penagos Molina and Max Rose and Antonios Pelekanidis and Wilhelm Eschen and J\"{u}rgen Hess and Thomas Wilhein and Rainer Heintzmann and Jan Rothhardt and Stefan Witte},
        journal = {Opt. Express},
        number = {9},
        pages = {13763--13797},
        publisher = {Optica Publishing Group},
        title = {PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox for conventional and Fourier ptychography},
        volume = {31},
        month = {Apr},
        year = {2023},
        doi = {10.1364/OE.485370},
}
```

