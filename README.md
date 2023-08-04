# PtyLab

This repository contains the Python implementation.

## Organisation of this repository:

- PtyLab contains the python module that enables reconstructions
- in example_scripts you can find typical use cases
- test contains all the tests of PtyLab
 
## Getting Started

To get started, clone this package and go to the root folder

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
```

To install this package and its dependencies in editable mode and in a virtual environment, for example using [virtualenv](https://pypi.org/project/virtualenv/), please do the following

```bash
virtualenv .venv
source .venv/bin/activate
pip install -e .
```

To use optional GPU utilities, please do the following

```bash
pip install -e .[gpu]
```

- If you are using Spyder click  ` Run-> Configuration per file -> Command line options: develop`
- Try running example scripts in the `example_scripts` folder 

