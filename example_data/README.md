# Simulation data

This folder contains small datasets that are mainly used to check for obvious errors in the reconstruction algorithms.

So far, it contains the following datasets:


**`ptycho_simulation.hdf5`**
Ptychograpgy simulation using 51 scan positions, the file contains following variables:  
*  ptychogram: 51\*128\*128, single  
*  positions: 51\*2, scan positions in k-space, single  
*  dxd: detector pixel in meter, single  
*  Nd: number of detector pixels, single  
*  No: number of object pixels, single  
*  Np: number of probe pixels, single  
*  zo: object camera distance, single  
*  wavelength: in meter, single
*  entrancePupilDiameter: defines the aperture diameter in k-space units, single


**`simulationTiny.hdf5`**
A tiny simulation of a binary object (try if you can reconstruct :)), the file contains following variables:  
*  ptychogram: 22\*64\*64, single  
*  probe: 64*64, complex single, helps reconstruction with known probe
*  xd: detector pixel in meter, single  
*  Nd: number of detector pixels (both in x,y), single  
*  zo: object camera distance, single  
*  wavelength: in meter, single  D
*  positions: scan positions in meter, single  
 



## Fourier ptychography data
**`LungCarcinomaFPM.hdf5`**
** `USAFTargetFPM.hdf5`**
Real Fourier ptychography dataset of a lung carcinoma sample and the USAF resolution test target.
*  ptychogram: 81\*128\*128, single  
*  dxd: detector pixel size in meter, single  
*  wavelength: in meter, single  
*  encoder: 81 scan positions on the led grid in meter, single  
*  zled: distance from the LED array to the sample, single
*  magnification: magnification of the optical system, single
Optional params:
*  NA: numerical aperture, single

# Loading

You can load the files in two ways: 

Either directly by using readExample:

```python
from fracPy.io import readExample

simulationData = readExample('simulation_tiny')
```

Or by creating a data object and augmenting the string with `example:`.

```python
from fracPy.FixedData.DefaultExperimentalData import ExperimentalData

loaderObject = ExperimentalData('example:fpm_dataset')
```
