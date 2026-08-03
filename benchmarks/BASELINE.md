# Performance baseline

Recorded before any optimization work, as the reference every later change is
measured against. Nothing here has been optimized yet; these are the numbers as
the library stands.

**Hardware:** NVIDIA H200 NVL, 139.8 GB, CUDA 12.9, CuPy 14.0.1, NumPy 2.4.3,
Python 3.13. Measured at PtyLab 0.2.8.

**Reproduce:** `python benchmarks/bench_engines.py --markdown`

Numbers move a few percent run to run. Treat anything under ~5% as noise.

## Regimes

Which optimization pays depends on the field size, so it is worth naming the
regimes rather than talking about "the" performance of the library:

| | field size | bound by | where the headroom is |
|---|---|---|---|
| A | ≤ ~12 MB | host dispatch / kernel launch | removing syncs, fusion, CUDA graphs |
| B | ≥ ~50 MB | HBM bandwidth + cuFFT | very little |
| C | OPR | orthogonalization time and memory | the linear algebra |

`field MB` = `nlambda × nosm × npsm × nslice × Np² × 8 bytes`.

## Engine baseline

3 timed iterations after a warm-up iteration, GPU backend.

| config | reg | field MB | ms/iter | µs/pos | peak GB |
|---|---|---|---|---|---|
| simu-like 128, 100fr | A | 0.1 | 58.6 | 586 | 0.03 |
| USAF-like 364, 102fr | A | 1.0 | 71.4 | 700 | 0.23 |
| mixed npsm=4 364 | A | 4.0 | 72.5 | 711 | 0.26 |
| Brain-like poly=7 182 | A | 1.8 | 74.9 | 749 | 0.06 |
| multislice nslice=4 364 | A | 4.0 | 150.6 | 1477 | 0.26 |
| heavy 7×2×4 364 | B | 56.6 | 90.0 | 2249 | 1.02 |
| **OPR 364, 202fr, 4 modes** | C | 4.0 | **505.4** | **2502** | **2.85** |

**The regime-A rows are flat at 586-749 µs/position across a 40x range of field
size** (0.1 → 4.0 MB). That flatness is the finding: the per-position cost barely
depends on how much data is being processed, which means the GPU is not the
thing being waited on. The cost is host-side dispatch.

OPR is 7x the cost of the comparable non-OPR row at the same field size
(505.4 vs 72.5 ms/iter at 4.0 MB) and 11x the memory.

## OPR iteration breakdown (364 px, 202 frames, 4 OPR modes, subspace 4)

| stage | ms/iter | share |
|---|---|---|
| `orthogonalizeIncoherentModes` (`OPR.py:149`) | 199.9 | **39.6%** |
| `orthogonalizeProbeStack` (`OPR.py:204`) | 133.8 | 26.5% |
| position loop and everything else | 170.8 | 33.9% |

**Two thirds of an OPR iteration is linear algebra, not ptychography.**
`orthogonalizeIncoherentModes` — a Python loop running one small SVD per frame —
is the larger of the two.

This matters for small cards. `probe_stack` is
`(1, 1, nModes, 1, Np, Np, nFrames)` complex64, so it grows linearly in frames
and modes: 0.80 GB at 364/202/4, 10.43 GB at 512/890/6, before the transient the
SVD itself needs on top.

## Negative results — measured, do not re-investigate

**FFT-size padding is not worth it.** Np=364 (`2²·7·13`) and Np=182 (`2·7·13`)
look like awkward cuFFT sizes, but measure at 2.6 µs/MB against 1.9 µs/MB for
N=1024 — only 1.4x off best-case, not the several-fold penalty a genuinely bad
size would show. Padding 364→384 measured **0.93x, i.e. slower**; 364→512 was
0.65x. Padding 182→192 does give 1.16x, but costs 11% more memory everywhere.

**Batching the existing SVD buys nothing.** Replacing the per-frame Python loop
in `orthogonalizeIncoherentModes` with a single batched `cp.linalg.svd` measured
1.02x at 364/202/4 and 1.03x at 512/890/6. The cost is inside the SVD itself,
not in per-call launch overhead, so batching the same algorithm does not help.
Only changing the algorithm does.

## Environment note

`cupy.linalg.eigh` **does not work in this environment**: `cupyx.cusolver` fails
to import with `libcusolver.so.11: cannot open shared object file`, while
`cupy.linalg.svd` works normally. This affects any user code calling `eigh`, and
it rules out eigendecomposition-based approaches here until the CUDA
installation is repaired.

## Test-suite baseline

The safety net these benchmarks are measured against.

| | before | now (0.2.8) |
|---|---|---|
| passing | 14 | 61 |
| skipped | 19 | 19 |
| engine configurations with a numerical golden | 0 | 9 |
| propagators with an output golden | 0 | 8 |
| runtime | 1.6 s | 1.9 s |

Verified that the goldens reproduce bit-exactly and that a 0.1% change to
`betaObject` is caught.
