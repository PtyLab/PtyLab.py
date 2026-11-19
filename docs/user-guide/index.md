# User Guide

How PtyLab.py works: the two ptychography modes, the input data, the engines and their settings.

- [**CPM Overview**](../cpm/overview.md) Conventional ptychography and how a CPM reconstruction is set up.
- [**FPM Overview**](../fpm/overview.md) Fourier ptychography with an LED array and how it maps onto the same framework.
- [**Data Format**](../cpm/data-format.md) The HDF5 fields PtyLab.py reads, with types and units.
- [**Engines**](../cpm/engines.md) The available reconstruction algorithms and when to use each.
- [**Parameter Reference**](../cpm/configuration.md) Every `Params` setting shared by the engines.
- [**Differentiable Ptychography**](../differentiable/overview.md) Gradient-based reconstruction with the optional PyTorch `GradientEngine`. *Work in progress.*

!!! tip "Learn by example"
    The [Examples](../examples/index.md) run these pieces end-to-end on simulated and real data. The [API Reference](../api/index.md) documents each class and function.
