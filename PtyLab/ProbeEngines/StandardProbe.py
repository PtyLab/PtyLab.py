import logging
import numpy as np

from PtyLab.utils.gpuUtils import isGpuArray, getArrayModule


class LinearProbe:
    def __init__(self):
        self.logger = logging.getLogger("SHG")
        self.probe = None
        self.probe_temp = None

    def clear(self):
        pass

    def push(self, new_probe, index, N_positions, factor=1.0, force=False):
        """
        Set the current estimate of the probe to new_probe.
        """
        if force:
            self.probe = new_probe
        elif self.probe is not None:
            self.probe = new_probe * factor + (1 - factor) * self.probe
        else:
            self.probe = new_probe
        self.probe_temp = self.probe.copy()

    def set_temporary(self, probe):
        """These map to self.reconstruction.probe. Can be used for quick updates in the calculation of the probe.

        Once you're done, make it official by updating with push()"""
        self.probe_temp = probe

    def get_temporary(self):
        return self.probe_temp

    def get(self, index):
        return self.probe

    def roll(self, dy, dx):
        self.probe = self.probe_temp.copy()
        xp = getArrayModule(self.probe)
        self.probe = xp.roll(self.probe, (-dy, -dx), axis=(-2, -1))
        self.probe_temp = self.probe.copy()


class SHGProbe(LinearProbe):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("SHG")
        self.probe = None  # wavelength = wavelength  * self.nonlinearity
        self.nonlinearity = 2

    def clear(self):
        pass

    def push(self, new_probe_nonlinear, index, N_positions):
        """Gets the update of the nonlinear part"""
        if self.probe is not None:
            if isGpuArray(new_probe_nonlinear) and not isGpuArray(self.probe):
                import cupy as cp

                print(" Moving self.probe to GPU")
                self.probe = cp.array(self.probe)
            else:
                xp = getArrayModule(self.probe)
                new_probe_nonlinear = xp.array(new_probe_nonlinear)
        else:
            xp = getArrayModule(new_probe_nonlinear)

        # what's actually pushed is the second harmonic. We need to update that

        if self.probe is None:
            self.probe = new_probe_nonlinear * 0
        # try "newtons" method

        # Solve for the new estimate, and update the original estimate based on it
        new_probe_estimate = new_probe_nonlinear ** (1.0 / self.nonlinearity)

        diff = new_probe_estimate - self.probe
        self.probe += diff / (2 * self.nonlinearity)
        # diff = new_probe_nonlinear - self.probe ** self.nonlinearity
        if N_positions == -1:
            print(np.linalg.norm(self.probe**self.nonlinearity - new_probe_nonlinear))
        # update_fundamental = self.nonlinearity * diff
        # self.probe += update_fundamental
        self.probe_temp = self.probe.copy() ** self.nonlinearity

    def change_nonlinearity(self, nonlinearity):
        last_probe = self.get(None).copy()
        self.nonlinearity = nonlinearity
        self._push_hard(last_probe)

    def _push_hard(self, new_probe, number_of_iterations=50):
        xp = getArrayModule(self.probe)
        new_probe = xp.array(new_probe)
        for i in range(number_of_iterations):
            self.push(new_probe, None, -1)
            print(xp.linalg.norm(self.get(None) - new_probe))

    def get(self, index):
        return self.probe**self.nonlinearity

    def get_fundamental(self):
        return self.probe
