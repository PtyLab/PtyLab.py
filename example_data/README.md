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
**`lung_441leds_fpm.hdf5`**
Real Fourier ptychography dataset of a lung carcinoma sample.
*  ptychogram: 128\*128\*441, single  
*  probe: 128*128, complex single, helps reconstruction with known probe
*  xd: detector pixel size in meter, single  
*  dxd: sample pixel size in meter, single  
*  Nd: number of detector pixels (both in x,y), single  
*  wavelength: in meter, single  
*  positions: 441*2 scan positions in pixels, single  
*  encoder: 441*2 scan positions on the led grid in meter, single  
*  aperture_diameter: physical aperture size in meters, single
*  NA: numerical aperture, single
*  focal_length: focal length of the lens, single
*  led_dist_to_sample: distance from the LED array to the sample, single
*  u: distance from the sample to the lens, single
*  zo = v: distance from the lens to the detector, single


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