from scipy import ndimage

from PtyLab.utils.gpuUtils import getArrayModule
import logging

try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numpy as np


class OPRP_storage:
    def __init__(self, N_probes=5, correct_position=True):
        self.logger = logging.getLogger("OPRP")
        self.N_probes = N_probes
        self.correct_position = correct_position

    def clear(self):
        """Use this when the probe is changed, for instance number of probe modes is changed"""
        for name in ["probes", "probe_indices", "A", "s", "At"]:
            if hasattr(self, name):
                delattr(self, name)

    def push(self, probe, index, N_positions):
        self.xp = getArrayModule(probe)

        probe = probe * self.xp.exp(-1j * self.xp.angle(probe.sum()))

        if not hasattr(self, "probes"):
            self._prepare_probes(probe, N_positions)
            # if it's the first one, make it small so the updates are relatively important
            probe_norm = probe.reshape(self.new_probe_shape)
            probe_norm = probe_norm / self.xp.linalg.norm(probe_norm)
            self.push(probe_norm.reshape(self.original_probe_shape), index, N_positions)
            return

        if self.correct_position:
            probe = self.center_probe(probe, index)

        self.probes[index] = probe.reshape(self.new_probe_shape)
        self.probe_indices[index] = True

    def tsvd(self):
        self.logger.info("Running TSVD")
        if not np.all(self.probe_indices == True):
            indices = self.xp.argwhere(self.probe_indices)
            probes = self.probes[indices]
        else:
            probes = self.probes
        average_power = self.xp.mean(self.xp.abs(probes**2))
        probe_power = self.xp.mean(self.xp.abs(probes**2), axis=-1, keepdims=True)
        probes *= self.xp.mean(average_power / (probe_power + 1e-6))

        A, s, At = self.xp.linalg.svd(probes, full_matrices=False)
        N = self.N_probes
        # calculate effective rank
        pk = s / self.xp.linalg.norm(s.flatten(), ord=1)
        H = -self.xp.sum(pk * self.xp.log(pk))
        eRank = self.xp.exp(H)

        self.A = A[:, :N]
        self.s = s[:N]
        self.At = At[:N]

        self.logger.info(
            f"Effective rank: {eRank}, truncating to {self.N_probes} modes"
        )

        self.logger.info(f"Average displacement: {np.mean(abs(self.center_mass))}")

    def get(self, index):
        """Get the TSVD estimate of the i-th index.

        If the particular index has not been given yet,
        or tsvd has not been run yet, return the averaged probe that we measured so far."""
        if not hasattr(self, "A"):
            # tsvd has not been run yet, return the averaged probe
            return (
                self.probes[self.probe_indices]
                .mean(axis=0)
                .reshape(self.original_probe_shape)
            )

        if self.probe_indices[index]:  # we measured this one, all easy
            A = self.A[index]
        else:  # tsvd has been run, but this particular probe was not given yet
            # beginning, we ask for a probe that we haven't measured yet.
            # In this case, return the typical probe to have some idea
            # Aka set self.A to [1, 0, 0,... 0]
            # we didn't measure it, return the first mode multiplied with the average power
            A = self.A[0].copy()
            A[1:] = 0
            A[0] = 1.0 * self.xp.sign(self.A[0, 0])
        probe = np.matmul(A, self.s[..., None] * self.At)
        # probe = (A * self.s) @ self.At
        # print(probe.shape)
        # move it back
        probe = probe.reshape(self.original_probe_shape)
        if self.correct_position:
            probe = self.uncenter_probe(probe, index)

        return probe

    def center_probe(self, probe, index):
        dpos = (
            np.array(ndimage.center_of_mass(abs(probe**2)))
            - np.array(probe.shape) / 2
        )
        dpos = np.clip(dpos, -2.5, 2.5)
        self.center_mass[index] += 0.01 * dpos
        # move it
        for dim, shift in enumerate(self.center_mass[index]):
            if self.original_probe_shape[dim] != 1:
                shift = np.round(shift).astype(int)
                probe = self.xp.roll(probe, shift=-shift, axis=dim)
        return probe

    def uncenter_probe(self, probe, index):
        probe = probe.reshape(self.original_probe_shape)
        for dim, shift in enumerate(self.center_mass[index]):
            if self.original_probe_shape[dim] != 1:
                shift = np.round(shift).astype(int)
                probe = self.xp.roll(probe, shift=shift, axis=dim)
        return probe

    def _prepare_probes(self, single_probe, N_positions):
        self.original_probe_shape = single_probe.shape
        probe_shape = np.array(single_probe.shape)
        # probe_shape[-2] = single_probe.shape[-1] * single_probe.shape[-2]
        # probe_shape = probe_shape[:-1]
        self.new_probe_shape = np.array([np.product(single_probe.shape)])
        # self.new_probe_shape = probe_shape
        self.N_positions = N_positions
        self.probes = self.xp.zeros(
            (self.N_positions, *self.new_probe_shape), dtype=np.complex64
        )
        self.probe_indices = self.xp.zeros(self.N_positions, dtype=np.bool)

        if self.correct_position:
            shape = (self.N_positions, len(self.original_probe_shape))
            self.center_mass = np.zeros(shape)

    def estimate_CM(self):
        from scipy import ndimage

        for i in range(self.N_positions):
            probe = self.get(i)
            cmass = np.array(ndimage.center_of_mass(abs(probe) ** 2))

            print(i, cmass - np.array(probe.shape) / 2 + 1)
