from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE, qNewton

# create an experimentalData object and load a measurement
exampleData = ExperimentalData()
# exampleData.loadData('example:simulationTiny')
# exampleData.loadData('example:fpm_dataset')
exampleData.loadData('example:simulation_fpm')
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.


# now create an object to hold everything we're eventually interested in
optimizable = Optimizable(exampleData)
# this will copy any attributes from experimental data that we might care to optimize

# now we want to run an optimizer. First create it.
qNewton_engine = qNewton.qNewton(optimizable, exampleData)
# set any settings involving ePIE in this object.
qNewton_engine.numIterations = 20
# now, run the reconstruction
qNewton_engine.doReconstruction()

# now we want to run an optimizer. First create it.
# ePIE_engine = ePIE.ePIE(optimizable, exampleData)
# # set any settings involving ePIE in this object.
# ePIE_engine.numIterations = 3
# # now, run the reconstruction
# ePIE_engine.doReconstruction()

# # OK, now I want to run a different reconstruction algorithm, what now?
# # just create a new reconstruction ePIE_engine
# # we initialize it with our data and optimizable parameters so we can forget about them afterwards
# mPIE_engine = mPIE.mPIE(optimizable, exampleData)

# mPIE_engine.doReconstruction()

# now save the data
# optimizable.saveResults('reconstruction.hdf5')
