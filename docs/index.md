# PtyLab.py

!!! warning "Work in Progress"
    This documentation is under active development. Some details may be incomplete or vary from the current state of the codebase. Report an [issue](https://github.com/PtyLab/PtyLab.py/issues) if you find inconsistencies.

PtyLab.py is an open-source inverse modeling toolbox for **Conventional Ptychographic Microscopy (CPM)** and **Fourier Ptychographic Microscopy (FPM)** in a unified framework. It supports a wide range of reconstruction engines, advanced correction algorithms, and transparent GPU acceleration.

For the original publication, see: [PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox for conventional and Fourier ptychography](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).

## Key features

- **Classic engines**: ePIE, mPIE, mqNewton, qNewton
- **Advanced corrections**: position correction (pcPIE), defocus correction (zPIE), angle correction (aPIE), orthogonal probe relaxation (OPR)
- **Multi-modal**: multi-slice, multi-wavelength, mixed-state object and probe
- **Multiple propagators**: Fraunhofer, Fresnel, Angular Spectrum (ASP), scaled ASP, polychromatic variants
- **GPU acceleration**: same code runs on CPU and GPU

## Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PtyLab/PtyLab.py/blob/main/demo.ipynb)

## Where to start

- **[Installation](getting-started/installation.md)** — Set up PtyLab.py with CPU or GPU support
- **[Quick Start](getting-started/quickstart.md)** — Run your first reconstruction in a few lines of code
- **[CPM Workflow](cpm/overview.md)** — Understand the conventional ptychography pipeline
- **[FPM Workflow](fpm/overview.md)** — Fourier ptychography with LED array illumination
- **[Tutorials](tutorials/tutorial_CPM_sim.ipynb)** — Jupyter notebooks covering common use cases

## Citation

If you use PtyLab.py in your work, please cite:

```bibtex
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
