# PtyLab

This repository contains the Python implementation of PtyLab. 
 
## Getting Started

Under [example_scripts](example_scripts/) you can find typical use cases of using PtyLab for your reconstruction. 

### Installation

To install the most recent PtyLab package from source and run on CPU

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git@main
```

If you want to utilize GPU for faster reconstruction, please install `cupy` based on the version of your CUDA driver. Check the [installation instructions](https://docs.cupy.dev/en/latest/install.html#installing-cupy-from-pypi) from their webpage. 

For example if your CUDA version is between v11.2 - v11.8,

```bash
pip install cupy-cuda11x
```

### Development

To get started, clone this package and go to the root folder

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
```

To install this package and its dependencies in editable mode and in a virtual environment, for example using [virtualenv](https://pypi.org/project/virtualenv/), please do the following

```bash
virtualenv .venv
source .venv/bin/activate
pip install -e .[dev]
```

To install `cupy` in your virtual environment based on your CUDA driver, please refer to the same [link](https://docs.cupy.dev/en/latest/install.html#installing-cupy-from-pypi) mentioned above.


