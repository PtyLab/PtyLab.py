# from fracPy import engines
#
#
# if __name__ == '__main__':
#     # This file is an example of what a reconstruction would look like from a user perspective
#     datafolder = '/tmp/data/'
#
#     reconstructor = engines.ePIE.ePIE(datafolder)
#
#     reconstructor.prepare_reconstruction()
#     # oops, forgot to set the wavelength
#     reconstructor.wavelength = 1064e-9
#     # do the reconstruction
#     reconstructor.reconstruct()


from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE


# create an experimentalData object and load a measurement
experimentalData = ExperimentalData()
experimentalData.loadData('my_awesome_measurement.hdf5')
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.


# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(experimentalData)
# this will copy any attributes from experimental data that we might care to optimize

# now we want to run an optimizer. First create it.
ePIE_engine = ePIE.ePIE(optimizable, experimentalData)
# set any settings involving ePIE in this object.
ePIE_engine.numIterations = 1
# now, run the reconstruction
ePIE_engine.reconstruct()

# OK, now I want to run a different reconstruction algorithm, what now?
# just create a new reconstruction ePIE_engine
# we initialize it with our data and optimizable parameters so we can forget about them afterwards
mPIE_engine = mPIE.mPIE(optimizable, experimentalData)

mPIE_engine.reconstruct()

# now save the data
optimizable.saveResults('reconstruction.hdf5')
