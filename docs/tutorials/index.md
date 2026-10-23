# Tutorials

Hands-on tutorials covering CPM and FPM reconstruction from simulation and real data.

## In this section

- [**01 – CPM Simulation**](tutorial_CPM_sim.md) — Simulate a CPM dataset and reconstruct it end-to-end.
- [**02 – CPM Example Data**](tutorial_CPM.md) — Reconstruct a real CPM dataset using `easyInitialize`.
- [**03 – FPM Example Data**](tutorial_FPM.md) — Reconstruct a Fourier ptychography dataset from LED-array images.
- [**04 – CPM Mixed States**](tutorial_mixedStates.md) — Reconstruct partially coherent X-ray data with several probe modes.
- [**05 – CPM Multi-Slice**](tutorial_multiSlice.md) — Reconstruct the layers of a thick sample separately with `e3PIE`.
- [**06 – CPM Multi-Wavelength**](tutorial_CPM_multiWavelength.md) — Handle datasets acquired with multiple illumination wavelengths.
- [**07 – CPM Position Correction**](tutorial_LateralPositionCorrection.md) — Correct lateral probe-position errors during reconstruction.
- [**08 – CPM Purity-Based Calibration**](tutorial_purityCalibration.md) — Calibrate the sample-to-detector distance or the wavelength from the probe purity.
- [**09 – CPM Reflection**](tutorial_reflectionPtychography.md) — Correct the tilted plane, calibrate the tilt angle with aPIE, and reconstruct reflection data.
- [**10 – CPM Interferometric Intensity**](tutorial_interferometricIntensityConstraint.md) — Apply an interferometric intensity constraint during reconstruction.
- [**11 – Scan Grids**](tutorial_scanGrids.md) — Generate scan grids, check their step size and optimize the scan path.
- [**12 – CPM Orthogonal Probe Relaxation**](tutorial_OPR.md) — Let the probe vary across the scan to handle a drifting beam.

## Running a tutorial yourself

Every tutorial page has buttons to download it as a Jupyter notebook or as a Python script. Run the script with `python tutorial_CPM_sim.py`. The scripts are also in [`docs/tutorials/scripts/`](https://github.com/PtyLab/PtyLab.py/tree/main/docs/tutorials/scripts) in the repository.
