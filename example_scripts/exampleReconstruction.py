import matplotlib
matplotlib.use('tkagg')
from fracPy.io import getExampleDataFolder


#matplotlib.use('qt5agg')
from fracPy.ExperimentalData.ExperimentalData import ExperimentalData
from fracPy.Optimizable.Optimizable import Optimizable
from fracPy.engines import ePIE, mPIE, qNewton, zPIE, e3PIE
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

FPM_simulation = False
ptycho_simulation = True


if FPM_simulation:
    # create an experimentalData object and load a measurement
    exampleData = ExperimentalData()
    exampleData.loadData('example:simulation_fpm')
    # exampleData.loadData('example:simulation_ptycho')
    exampleData.operationMode = 'FPM'
    # # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
    # # now create an object to hold everything we're eventually interested in
    optimizable = Optimizable(exampleData)

    optimizable.npsm = 9  # Number of probe modes to reconstruct
    optimizable.nosm = 1  # Number of object modes to reconstruct
    optimizable.nlambda = 1  # Number of wavelength
    optimizable.prepare_reconstruction()
    # # this will copy any attributes from experimental data that we might care to optimize

    # Set monitor properties
    monitor = Monitor()
    monitor.figureUpdateFrequency = 1
    monitor.objectPlot = 'complex'
    monitor.verboseLevel = 'high'

    # now we want to run an optimizer. First create it.
    # qNewton_engine = qNewton.qNewton(optimizable, exampleData, monitor)
    qNewton_engine = qNewton.qNewton_GPU(optimizable, exampleData, monitor)
    # set any settings involving ePIE in this object.
    qNewton_engine.numIterations = 50
    # now, run the reconstruction
    qNewton_engine.doReconstruction()


""" 
ptycho data reconstructor 
change data visualization and initialization options manually for now
"""
if ptycho_simulation:

    exampleData = ExperimentalData()

    import os
    filePath = 'WFSpoly.hdf5'
    filePath = getExampleDataFolder() / filePath

    #os.chdir(filePath)

    exampleData.loadData(filePath)  # simuRecent  Lenspaper

    exampleData.operationMode = 'CPM'
    # M = (1+np.sqrt(1-4*exampleData.dxo/exampleData.dxd)/2*exampleData.dxo/exampleData.dxd)
    # exampleData.zo = exampleData.zo/M
    # exampleData.dxd = exampleData.dxd/M
    # absorbedPhase = np.exp(1.j*np.pi/exampleData.wavelength *
    #                                              (exampleData.Xp**2+exampleData.Yp**2)/(exampleData.zo))
    # absorbedPhase2 = np.exp(1.j*np.pi/exampleData.wavelength *
    #                                              (exampleData.Xp**2+exampleData.Yp**2)/(exampleData.zo))

    # now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
    # now create an object to hold everything we're eventually interested in
    optimizable = Optimizable(exampleData)
    optimizable.npsm = 1 # Number of probe modes to reconstruct
    optimizable.nosm = 1 # Number of object modes to reconstruct
    optimizable.nlambda = 1 # Number of wavelength
    optimizable.nslice = 1 # Number of object slice
    exampleData.dz = 1e-4  # slice
    exampleData.dxp = exampleData.dxd


    optimizable.initialProbe = 'circ'
    exampleData.entrancePupilDiameter = exampleData.Np / 3 * exampleData.dxp  # initial estimate of beam
    optimizable.initialObject = 'ones'
    # initialize probe and object and related params
    optimizable.prepare_reconstruction()

    # customize initial probe quadratic phase
    # optimizable.probe = optimizable.probe*np.exp(1.j*2*np.pi/exampleData.wavelength *
    #                                              (exampleData.Xp**2+exampleData.Yp**2)/(2*6e-3))
    from fracPy.utils.utils import rect, fft2c, ifft2c
    from fracPy.utils.scanGrids import GenerateRasterGrid
    pinholeDiameter = 730e-6
    fullPeriod = 6 * 13.5e-6
    apertureSize = 4 * 13.5e-6
    WFS = 0 * exampleData.Xp
    n = int(pinholeDiameter // fullPeriod)
    R, C = GenerateRasterGrid(n, np.round(fullPeriod / exampleData.dxp))
    print('WFS size: %d um' % (2 * max(max(np.abs(C)), max(np.abs(R))) * exampleData.dxp * 1e6))
    print('WFS size: %d um' % ((max(max(R) - min(R), max(C) - min(C)) * exampleData.dxp + apertureSize) * 1e6))
    R = R + exampleData.Np // 2
    C = C + exampleData.Np // 2

    np.random.seed(1)
    R_offset = np.random.randint(1, 3, len(R))
    np.random.seed(2)
    C_offset = np.random.randint(1, 3, len(C))
    R = R + R_offset - 2
    C = C + C_offset - 2

    for k in np.arange(len(R)):
        WFS[R[k], C[k]] = 1

    subaperture = rect(exampleData.Xp / apertureSize) * rect(exampleData.Yp / apertureSize)
    WFS = np.abs(ifft2c(fft2c(WFS) * fft2c(subaperture)))  # convolution of the subaperture with the scan grid
    WFS = WFS / np.max(WFS)

    optimizable.probe[..., :, :] = WFS.astype('complex64')
    hsvplot(np.squeeze(optimizable.probe), pixelSize=exampleData.dxp, axisUnit='mm')
    plt.show(block=False)

    # this will copy any attributes from experimental data that we might care to optimize
    # # Set monitor properties
    monitor = Monitor()
    monitor.figureUpdateFrequency = 1
    monitor.objectPlot = 'complex'  # complex abs angle
    monitor.verboseLevel = 'low'  # high: plot two figures, low: plot only one figure

    exampleData.zo = exampleData.zo
    exampleData.spectralDensity = [exampleData.wavelength]
    # exampleData.dxp = exampleData.dxp/1
    # Run the reconstruction
    ## choose engine
    # ePIE
    # engine = ePIE.ePIE_GPU(optimizable, exampleData, monitor)
    # engine = ePIE.ePIE(optimizable, exampleData, monitor)
    # mPIE
    engine = mPIE.mPIE_GPU(optimizable, exampleData, monitor)
    # engine = mPIE.mPIE(optimizable, exampleData, monitor)
    # zPIE
    # engine = zPIE.zPIE_GPU(optimizable, exampleData, monitor)
    # engine = zPIE.zPIE(optimizable, exampleData, monitor)
    # engine.zPIEgradientStepSize = 200
    # e3PIE
    # engine = e3PIE.e3PIE_GPU(optimizable, exampleData, monitor)
    # engine = e3PIE.e3PIE(optimizable, exampleData, monitor)

    ## main parameters
    engine.numIterations = 100
    engine.positionOrder = 'random'  # 'sequential' or 'random'
    engine.propagator = 'ASP'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP
    engine.betaProbe = 0.0
    engine.betaObject = 0.25

    ## engine specific parameters:
    engine.zPIEgradientStepSize = 100  # gradient step size for axial position correction (typical range [1, 100])

    ## switches
    engine.probePowerCorrectionSwitch = True
    engine.modulusEnforcedProbeSwitch = False
    engine.comStabilizationSwitch = True
    engine.orthogonalizationSwitch = True
    engine.orthogonalizationFrequency = 10
    engine.fftshiftSwitch = False
    engine.intensityConstraint = 'standard'  # standard fluctuation exponential poission
    engine.absorbingProbeBoundary = False
    engine.objectContrastSwitch = False
    engine.absObjectSwitch = False
    engine.backgroundModeSwitch = False

    engine.doReconstruction()


    # now save the data
    # optimizable.saveResults('reconstruction.hdf5')
