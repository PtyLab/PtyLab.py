# PtyLab.py: Unified Ptychography Toolbox
![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)
[![PyPI](https://img.shields.io/pypi/v/ptylab.svg)](https://pypi.org/project/ptylab/)
[![Downloads](https://img.shields.io/pypi/dm/ptylab.svg)](https://pypi.org/project/ptylab/)
![Tests](https://github.com/PtyLab/PtyLab.py/actions/workflows/test.yml/badge.svg)

[**Getting Started**](#getting-started) | [**Installation**](#installation) | [**Development**](#development) | [**Documentation**](https://ptylab.github.io/PtyLab.py/)

PtyLab is an inverse modeling toolbox for Conventional (CP) and Fourier (FP) ptychography in a unified framework. For more information please check the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).

## Key Features

- **Classic engines**: ePIE, mPIE, mqNewton, qNewton
- **Advanced corrections**: position correction (pcPIE), defocus correction (zPIE), angle correction (aPIE), orthogonal probe relaxation (OPR)
- **Multi-modal**: multi-slice, multi-wavelength, mixed-state object and probe
- **Multiple propagators**: Fraunhofer, Fresnel, Angular Spectrum (ASP), scaled ASP, polychromatic variants
- **GPU acceleration**: same code runs on CPU and GPU

The reconstructed output is a 6D array of shape `(nlambda, nosm, npsm, nslice, No, No)`:

| Dim | Meaning |
|-----|---------|
| `nlambda` | wavelengths |
| `nosm` | object state mixture |
| `npsm` | probe state mixture |
| `nslice` | depth slices |
| `No` | output frame size |


## Getting started

Try the demo in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PtyLab/PtyLab.py/blob/main/demo.ipynb)
![demo](https://raw.githubusercontent.com/PtyLab/PtyLab.py/main/assets/recon.gif)

For more use cases, start with the [tutorials](https://ptylab.github.io/PtyLab.py/tutorials/) in our documentation, then see the [example_scripts](https://github.com/PtyLab/PtyLab.py/tree/main/example_scripts) directory.

## Installation

Install from PyPI within your virtual environment:

```bash
pip install ptylab
```

> [!NOTE]  
> For much faster installs, we recommend [uv](https://docs.astral.sh/uv/getting-started/installation/): `uv pip install ptylab`

This package uses `cupy` to utilize GPU for faster reconstruction. To enable GPU support:

```bash
pip install "ptylab[gpu]"
```

For the latest unreleased changes on `main`:

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git
```

Check GPU detection:

```bash
ptylab check gpu
```

### Development

Clone the repo and install dev dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
uv sync --extra dev
```

This creates a `.venv` virtual environment in the project root. Select this environment from your IDE.

To use the GPU as well, install with the `gpu` flag:

```bash
uv sync --extra dev,gpu
```

and check if GPU is detected with `uv run ptylab check gpu`.

Add tests for new behaviour and run the suite:

```bash
uv run pytest tests
```

CI runs this on every PR. Please bump the package version when modifying dependencies.

Releases are published to PyPI automatically: bump `version` in `pyproject.toml`, then publish a GitHub Release tagged `v<version>`.

## Citation

If you use this package, please cite:

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

