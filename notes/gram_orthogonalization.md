# The "gram" orthogonalization in OPR — what it is and why it is faster

This explains the change introduced in `PtyLab/Engines/OPR.py` behind
`params.OPR_tsvd_type = "gram"` and `params.OPR_fast_orthogonalization = True`.
It replaces two SVD calls with a Gram-matrix factorization. Nothing about the
physics changes; it is a different route to (almost) the same numbers.

Measured on the H200, OPR at Np=364, 202 frames, 4 modes:

| stage | before (full SVD) | after (gram) | speedup |
|---|---|---|---|
| `orthogonalizeIncoherentModes` | 201.1 ms | 48.6 ms | **4.1x** |
| `orthogonalizeProbeStack` | 139.0 ms | 77.0 ms | **1.8x** |
| position loop + everything else | 159.1 ms | 160.1 ms | 1.0x |
| **total per iteration** | **499.2 ms** | **285.7 ms** | **1.75x** |

Peak GPU memory over a 30-iteration run drops from 3.18 GB to 1.85 GB (1.72x).

---

## 1. The background: what an SVD gives you, and what OPR actually needs

Any matrix `A` factors as

$$
A = U S V^{H}
$$

with `U` and `V` having orthonormal columns and `S` diagonal with the singular
values in descending order. The expensive part on a GPU is that this is computed
by an iterative bidiagonalization — a long sequence of small, poorly-parallel
steps. It is the opposite of what a GPU is good at.

The key observation is that **OPR never needs `U`, `S` and `V` separately.** It
needs two specific products, and both can be obtained much more cheaply.

---

## 2. The core identity

Form the **Gram matrix** of `A`:

$$
G = A^{H} A = (U S V^{H})^{H} (U S V^{H}) = V S U^{H} U S V^{H} = V S^{2} V^{H}
$$

(using $U^{H}U = I$). So `G`'s eigenvectors are exactly `A`'s right singular
vectors `V`, and its eigenvalues are the **squared** singular values. Equally,

$$
A A^{H} = U S^{2} U^{H}
$$

gives the left singular vectors. So you can recover whichever side you need from
a decomposition of a *much smaller* matrix — and the trick is to form the Gram
matrix on whichever side is small.

---

## 3. Use one: `orthogonalizeIncoherentModes` — the 4.1x

**What it computes.** For each scan position, take the probe's incoherent modes
as a matrix `P` of shape `(nModes, Np²)` — for a typical run that is
**`(4, 132496)`**: 4 rows, 132 thousand columns. It orthogonalizes them and
rescales by mode power, i.e. it wants $S V^{H}$.

**The old way.** A full SVD of `(4, 132496)`, inside a Python `for` loop over all
202 frames. The economy SVD's $V^{H}$ has shape `(4, 132496)` — as large as the
input — so you pay a full factorization to produce something the size of what
you started with, 202 separate times.

**The new way.** Note that

$$
S V^{H} = U^{H} A
\qquad \text{since } A = U S V^{H} \text{ and } U^{H} U = I
$$

and `U` comes from the *other* Gram matrix:

$$
G = P P^{H}
\qquad \text{shape } (4, 4)
$$

Four by four. So the whole operation becomes: one small matmul to build a 4×4 matrix,
decompose that 4×4, then one matmul $U^{H} P$. The large dimension (132496) is only
ever touched by matrix multiplications, which run at near peak throughput.

On top of that, all 202 frames are done in **one batched call** instead of a
Python loop, so the per-frame launch overhead disappears too.

> **Why not just batch the original SVD?** I measured that: batching
> `cupy.linalg.svd` over frames gives **1.02x** — nothing. The cost is inside the
> SVD algorithm, not in the per-call overhead. Only changing the algorithm helps.

---

## 4. Use two: `orthogonalizeProbeStack` — the 1.8x

**What it computes.** For each OPR mode, the probes across all frames form a
matrix `A` of shape `(Np², nFrames)` — typically **`(132496, 202)`**: very tall
and thin. OPR projects this onto its best rank-`k` approximation (`k` =
`OPR_subspace`, usually 3–6). That is the "only a few degrees of freedom explain
how the probe varies across the scan" assumption.

**The old way.** Full SVD of the tall matrix, then zero all singular values past
`k` and multiply back together. This allocates a `U` of shape `(132496, 202)` —
about 204 MB in complex64 — purely to throw most of it away.

**The new way.** Build

$$
G = A^{H} A
\qquad \text{shape } (202, 202)
$$

which is `nFrames` squared, and therefore tiny. Decompose it, keep the top `k`
eigenvectors $V_{k}$, and the rank-`k` truncation is

$$
A_{k} = A V_{k} V_{k}^{H}
$$

`U` is never formed at full size. `G` at 202×202 is well under a megabyte,
versus 204 MB for the `U` the full SVD insists on building.

### Why is this faster if both are $O(M N^{2})$?

Forming `G` costs roughly the same *flop count* as the SVD. The difference is
what those flops are:

- $A^{H} A$ is a single **GEMM** — the most optimized operation on the machine,
  running at a large fraction of peak throughput.
- An SVD is a long chain of Householder reflections and QR sweeps: many small
  kernels, limited parallelism, lots of synchronization.

Then the remaining $202^{3} \approx 8 \times 10^{6}$ flops to decompose $G$ are
negligible next to the $132496 \times 202^{2} \approx 5 \times 10^{9}$ in the GEMM.

So the speedup comes from **moving the work from a badly-parallel algorithm into
a well-parallel one**, not from doing asymptotically less arithmetic. The
advantage is largest exactly when the matrix is very tall and thin, which is the
OPR regime.

---

## 5. What is preserved exactly, and what is not

This matters, so it is worth being precise.

**Preserved:**
- The singular values / mode powers — measured agreement 2.4e-7.
- The rank-k *subspace* spanned by the retained components.
- The rank-k truncation `A_k` itself — this is a **gauge-invariant** quantity
  (it does not depend on any phase convention), so `orthogonalizeProbeStack` is
  a faithful drop-in. Measured agreement 1.6e-5 to 3.3e-4.

**Not preserved bit-for-bit:**
- The *individual* singular vectors. Eigenvectors are only defined up to a phase
  factor, and when two modes carry near-equal power the vectors within that
  shared subspace are not determined at all — any rotation of them is equally
  valid. LAPACK's SVD makes an arbitrary but deterministic choice; the Gram
  route makes a different arbitrary choice.

This only affects `orthogonalizeIncoherentModes`, which returns individual mode
vectors. In practice it measures at **3.1e-5** relative after 30 iterations,
which is negligible.

### Is the resulting difference acceptable?

The end-to-end object differs by **1.6e-2** relative after 30 OPR iterations,
which looks alarming until you calibrate it. Perturbing the *initial object* by a
relative **1e-6** — below float32 resolution of the inputs — and running the
completely unmodified baseline moves the final object by **3.1e-2**.

The reconstruction is a chaotic fixed-point iteration: it amplifies any
perturbation. The Gram route's deviation is *smaller* than what a last-bit change
to the input produces, so it sits inside the algorithm's own noise floor. The
error metric — the quantity that actually measures reconstruction quality —
agrees to **9.3e-5**.

---

## 6. Two implementation details worth knowing

### `G` is built in double precision

Forming $A^{H} A$ squares the condition number: if $A$ has condition number
$\kappa$, then $G$ has $\kappa^{2}$. In float32 (≈7 digits) that could leave only ~3.5 usable digits,
which is why the Gram trick has a reputation for being numerically sloppy.

The fix is cheap. In `gram_tsvd` the sensitive step is decomposing `G`, and `G`
is only `nFrames × nFrames` — so it is cast to complex128 and decomposed in
double precision at negligible cost, then the result is cast back. That is what
the `.astype(xp.complex128)` is doing.

The batched `orthogonalizeIncoherentModes` path deliberately does *not* do this:
there `G` is only 4×4 (or 6×6), far too small and well-conditioned for the
squared condition number to matter, and casting after the GEMM would not recover
precision lost inside it anyway.

### We call `svd` on `G`, not `eigh`

`G` is Hermitian positive semi-definite, so its SVD and its eigendecomposition
are the same thing — and `svd` conveniently returns them already sorted in
descending order, whereas `eigh` returns ascending.

The practical reason, though, is that **`cupy.linalg.eigh` is not usable in this
environment**: it routes through `cupyx.cusolver`, which fails to import with
`libcusolver.so.11: cannot open shared object file`. `cupy.linalg.svd` uses
CuPy's own bindings and works. Using `svd` avoids depending on a code path that
is not reliably present across CuPy/CUDA installations.

This is worth fixing in the environment separately — it affects any user code
calling `eigh` — but the implementation deliberately does not depend on it.

---

## 7. Turning it off

Both switches are independent, and setting them back reproduces the previous
numerics exactly:

```python
params.OPR_tsvd_type = "numpy"              # full SVD in orthogonalizeProbeStack
params.OPR_fast_orthogonalization = False   # per-frame loop in orthogonalizeIncoherentModes
```

`params.OPR_tsvd_type` also still accepts `"randomized"`, which uses the
pre-existing randomized SVD in `PtyLab/utils/fsvd.py`.

The reference path is pinned by
`tests/regression/test_opr_regression.py::test_opr_legacy_path_golden`, and
`test_opr_gram_matches_legacy_on_gauge_invariant_quantities` asserts the fast
paths track it.
`tests/Engines/test_opr_linalg.py` unit-tests `gram_tsvd` directly against
`numpy.linalg.svd`, including rank-deficient input and rank clamping, and runs on
CPU so CI covers it.

---

## 8. Where this does *not* help

- **The position loop is untouched** (159 → 160 ms above). That is a separate
  problem needing a separate fix: it is bound by host-side kernel dispatch, not
  by any of the linear algebra discussed here.
- **Short frame counts.** The benefit of avoiding a large `U` shrinks as
  `nFrames` approaches `Np²`. For OPR runs the matrix is always extremely tall,
  so this is not a practical concern, but the code keeps the `"numpy"` path for
  cases where it is.
- **Ill-conditioned stacks with no spectral gap.** If the retained and discarded
  singular values are not well separated, the rank-k split is ambiguous for
  *any* algorithm — the Gram route does not make this worse, but it does not fix
  it either.
