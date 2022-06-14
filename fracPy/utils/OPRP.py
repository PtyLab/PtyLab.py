try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numpy as np

class OPRP_storage():
    def __init__(self, N_positions, single_probe, use_GPU, N_probes=5):
        if use_GPU:
            self.xp = cp
        else:
            self.xp = np
        self.original_probe_shape = single_probe.shape
        probe_shape = np.array(single_probe.shape)

        probe_shape[-2] = single_probe.shape[-1]*single_probe.shape[-2]
        probe_shape = probe_shape[:-1]
        self.new_probe_shape = probe_shape

        self.probes = self.xp.zeros((N_positions, *self.new_probe_shape))
        self.N_probes = N_probes

    def push(self, probe, index):
        self.probes[index] = probe.reshape(self.new_probe_shape)

    def tsvd(self):
        A, s, At = self.xp.linalg.svd(self.probes, full_matrices=False)
        N = self.N_probes
        self.A = A[:, :N]
        self.s = s[:N]
        self.At = At[:N]

    def get(self, index):
        """Get the TSVD estimate of the i-th index """
        probe = (self.A[index]*self.s)@self.At
        return probe.reshape(self.original_probe_shape)
