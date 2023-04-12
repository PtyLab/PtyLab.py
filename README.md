# PtyLab

This repository contains the Python implementation.

## Organisation of this repository:

- PtyLab contains the python module that enables reconstructions
- in example_scripts you can find typical use cases
- test contains all the tests of PtyLab
 
## Getting Started
- Clone and download the repository
- run `setup.py` to install any dependencies with
```
pip install .
```
- If you are using Spyder click  ` Run-> Configuration per file -> Command line options: develop`
-  try running expample scripts in the `example_scripts` folder 

## Development mode
- Create a virtual environment using `environment.yml` file
```
conda env create -f environment.yml
```

- Install this package in development mode from `setup.py`
```
pip install -e .
```

