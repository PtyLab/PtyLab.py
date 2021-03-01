import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.Optimizable.CalibrationFPM import IlluminationCalibration
from fracPy.engines import ePIE, mPIE, qNewton, zPIE, e3PIE, pcPIE
from fracPy.utils.gpuUtils import getArrayModule, asNumpyArray
from fracPy.monitors.Monitor import Monitor as Monitor
import logging
logging.basicConfig(level=logging.INFO)
from fracPy.utils.utils import ifft2c
from fracPy.utils.visualisation import hsvplot
from matplotlib import pyplot as plt
import numpy as np

""" 
FPM data reconstructor 
change data visualization and initialization options manually for now
"""

FPM_recon = False
ptycho_recon = True


if FPM_recon:
    
    #%% Load in the data
    exampleData = ExperimentalData()
    exampleData.operationMode = 'FPM'

    fileName = 'lung_441leds_fpm.hdf5'  # experimental data
    # fileName = 'HeLa_49leds_fpm.hdf5'  # experimental data
    filePath = getExampleDataFolder() / fileName
    exampleData.loadData(filePath)
    
    # decide whether the positions will be recomputed each time they are called or whether they will be fixed
    # without the switch, positions are computed from the encoder values
    # with the switch calling exampleData.positions will return positions0
    exampleData.fixedPositions = True 


    #%% FPM position calibration
    # NOTE: exampleData.fixedPositions = True must be used!
    calib = IlluminationCalibration(exampleData)
    # calib.plot=True
    calib.fitMode = 'Translation'
    calibrated_positions, _, _ = calib.runCalibration()
    # update the fixed non-dynamically computed positions with the new ones
    exampleData.positions0 = calibrated_positions
    
    
    #%% Prepare everything for the reconstruction
    # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
    # now create an object to hold everything we're eventually interested in
    optimizable = Optimizable(exampleData)
    optimizable.initialProbe = 'circ'
    optimizable.initialObject = 'upsampled'
    optimizable.prepare_reconstruction()
   
    # Set monitor properties
    monitor = Monitor()
    monitor.figureUpdateFrequency = 1
    monitor.objectPlot = 'abs'  # complex abs angle
    monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
    monitor.objectPlotZoom = .01   # control object plot FoV
    monitor.probePlotZoom = .01   # control probe plot FoV

    #%% RUn the reconstructor
    # engine = ePIE.ePIE_GPU(optimizable, exampleData, monitor)
    engine = qNewton.qNewton_GPU(optimizable, exampleData, monitor)
    # engine = mPIE.mPIE_GPU(optimizable, exampleData, monitor)
    
    # engine.positionOrder = 'NA'
    engine.probePowerCorrectionSwitch = False 
    engine.comStabilizationSwitch = False
    engine.probeBoundary = True
    # engine.absorbingProbeBoundary = True
    engine.numIterations = 50
    engine.betaProbe = 1
    engine.betaObject = 1
    
    # now, run the reconstruction
    engine.doReconstruction()


#%%
""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""
if ptycho_recon:

    exampleData = ExperimentalData()

    import os
    fileName = 'simu_PC_09deviation.hdf5'  # WFSpoly   WFS_SingleWave  WFS_9Wave simuRecent  Lenspaper
    filePath = getExampleDataFolder() / fileName

    exampleData.loadData(filePath)

    exampleData.operationMode = 'CPM'
    
    # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
    # now create an object to hold everything we're eventually interested in
    optimizable = Optimizable(exampleData)
    optimizable.npsm = 1 # Number of probe modes to reconstruct
    optimizable.nosm = 1 # Number of object modes to reconstruct
    optimizable.nlambda = 1 # len(exampleData.spectralDensity) # Number of wavelength
    optimizable.nslice = 1 # Number of object slice
    exampleData.dz = 1e-4  # slice
    # exampleData.dxp = exampleData.dxd


    optimizable.initialProbe = 'circ'
    exampleData.entrancePupilDiameter = exampleData.Np / 3 * exampleData.dxp  # initial estimate of beam
    optimizable.initialObject = 'ones'
    # initialize probe and object and related params
    optimizable.prepare_reconstruction()

    # customize initial probe quadratic phase
    optimizable.probe = optimizable.probe*np.exp(1.j*2*np.pi/exampleData.wavelength *
                                                 (exampleData.Xp**2+exampleData.Yp**2)/(2*6e-3))

    # this will copy any attributes from experimental data that we might care to optimize
    # # Set monitor properties
    monitor = Monitor()
    monitor.figureUpdateFrequency = 10
    monitor.objectPlot = 'complex'  # complex abs angle
    monitor.verboseLevel = 'high'  # high: plot two figures, low: plot only one figure
    monitor.objectPlotZoom = 1.5   # control object plot FoV
    monitor.probePlotZoom = 0.5   # control probe plot FoV

    # exampleData.zo = exampleData.zo
    # exampleData.dxp = exampleData.dxp/1
    # Run the reconstruction
    ## choose engine
    # ePIE
    # engine = ePIE.ePIE_GPU(optimizable, exampleData, monitor)
    # engine = ePIE.ePIE(optimizable, exampleData, monitor)
    # mPIE
    # engine = mPIE.mPIE_GPU(optimizable, exampleData, monitor)
    # engine = mPIE.mPIE(optimizable, exampleData, monitor)
    # zPIE
    # engine = zPIE.zPIE_GPU(optimizable, exampleData, monitor)
    # engine = zPIE.zPIE(optimizable, exampleData, monitor)
    # e3PIE
    # engine = e3PIE.e3PIE_GPU(optimizable, exampleData, monitor)
    # engine = e3PIE.e3PIE(optimizable, exampleData, monitor)
    # pcPIE
    engine = pcPIE.pcPIE_GPU(optimizable, exampleData, monitor)
    # engine = pcPIE.pcPIE(optimizable, exampleData, monitor)

    ## main parameters
    engine.numIterations = 1000
    engine.positionOrder = 'random'  # 'sequential' or 'random'
    engine.propagator = 'Fresnel'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
    engine.betaProbe = 0.25
    engine.betaObject = 0.25

    ## engine specific parameters:
    engine.zPIEgradientStepSize = 200  # gradient step size for axial position correction (typical range [1, 100])

    ## switches
    engine.probePowerCorrectionSwitch = True
    engine.modulusEnforcedProbeSwitch = False
    engine.comStabilizationSwitch = True
    engine.orthogonalizationSwitch = False
    engine.orthogonalizationFrequency = 10
    engine.fftshiftSwitch = False
    engine.intensityConstraint = 'standard'  # standard fluctuation exponential poission
    engine.absorbingProbeBoundary = False
    engine.objectContrastSwitch = False
    engine.absObjectSwitch = False
    engine.backgroundModeSwitch = False
    engine.couplingSwitch = True
    engine.couplingAleph = 1
    engine.positionCorrectionSwitch = True

    engine.doReconstruction()


    # now save the data
    # optimizable.saveResults('reconstruction.hdf5')
