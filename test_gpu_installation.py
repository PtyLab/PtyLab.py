# This file is to check that the cuda installation has been performed correctly.
# Sometimes, for strange combinations of the cuda toolkit and the cupy package, we can run into trouble with larger SVD's.
# This is mainly a problem in OPR.

import cupy as cp

a = cp.random.rand(10, 6, 512, 513).astype(cp.complex64)
i,j,k,l = a.shape
a, v, at = cp.linalg.svd(a.reshape(i,j, k*l), full_matrices=False)
print(a.shape)
assert a.shape == (10, 6, 6)
print('cuda seems to be correctly installed')