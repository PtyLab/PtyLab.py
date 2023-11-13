# PtyLab

This repository contains the Python implementation of PtyLab. 
 
## Getting Started

Under [example_scripts](example_scripts/) you can find typical use cases of using PtyLab for your reconstruction. 

### Installation

To install the most recent PtyLab package from source and run on CPU as a default action,

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git@main
```

#### Installation with optional dependencies

This package uses `cupy` to utilize GPU for faster reconstruction. 

> [!WARNING]
> The system must have [CUDA 11.x or 12.x](https://docs.nvidia.com/cuda/#) installed. Please note the version before proceeding.
 
- For CUDA v11.2 - 11.8 (x86_64 / aarch64):
```bash
pip install ptylab[gpu11x]@git+https://github.com/PtyLab/PtyLab.py.git@main
```

- For CUDA v12.x (x86_64 / aarch64)
```bash
pip install ptylab[gpu12x]@git+https://github.com/PtyLab/PtyLab.py.git@main
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

If you want to utilize GPU with CUDA v11.2 - 11.8 installed in system, do the following instead,

```bash
pip install -e .[dev,gpu11x]  # `gpu12x` if CUDA v12.x
```
