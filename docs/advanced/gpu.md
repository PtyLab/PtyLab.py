# GPU Acceleration

PtyLab.py uses [CuPy](https://cupy.dev/) as a drop-in GPU backend for NumPy. The same reconstruction code runs on both CPU and GPU — no changes to your script needed beyond toggling a single parameter.

## Installation with GPU support

GPU acceleration requires installing PtyLab.py with a CUDA extra. See [Installation](../getting-started/installation.md) for the full instructions.

```bash
pip install "ptylab[gpu] @ git+https://github.com/PtyLab/PtyLab.py.git"
```

## Verifying GPU availability

From the command line (within your environment):

```bash
ptylab check gpu
```

## Toggle GPU switch

By default, GPU is automatically detected and used for your reconstructions, however this can be also manually controlled via the following parameter

```bash
params.gpuSwitch = True
```

## Memory optimization

For large datasets that do not fit entirely in GPU memory:

```python
params.saveMemory = True
```

This reduces peak GPU memory usage at a slight performance cost by keeping some arrays on the CPU and transferring them as needed.

## Single precision

Converting to single precision (float32) halves memory usage and can significantly speed up GPU reconstruction:

```python
engine.convert2single()
```

Call this after creating the engine but before `reconstruct()`.

## Manual GPU/CPU transfer

In advanced workflows you can control data placement explicitly:

```python
engine._move_data_to_gpu()    # move all arrays to GPU
engine._move_data_to_cpu()    # move all arrays back to CPU
```

## Notes

- GPU support is **optional** — PtyLab.py runs identically on CPU without CuPy installed.
- The `params.gpuSwitch` property validates that CuPy is available before enabling GPU mode. It will raise an informative error if CuPy is not installed.
- Results are automatically transferred back to CPU before saving with `reconstruction.saveResults()`.
