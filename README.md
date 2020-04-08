# fracPy

This repository contains the Python implementation.

Organisation of this repository:

 * fracPy contains the python module that enables reconstructions
 * in example_scripts you can find typical use cases
 * test contains all the tests of fracpy
 
To get started, clone the repository, run setup.py to install any dependencies, and try running examples/generate_data.py.

Known issues:
* Almost nothing is implemented yet.

SimuData.mat description:
ptychogram: 64*64*20 single, diffraction intensities
xd: detector pixel size in meter
Nd: Number of detector pixels in both x and y direction
zo: distance between the detector and object
wavelength: wavelength in meter
positions: 20 scan positions
probe: 64*64 complex single
