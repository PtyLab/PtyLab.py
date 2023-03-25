from unittest import TestCase
import unittest
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
import PtyLab

try:
    import cupy as cp
except ImportError:
    cp = None


class TestBaseEngine(TestCase):
    def setUp(self) -> None:
        (
            experimentalData,
            reconstruction,
            params,
            monitor,
            ePIE_engine,
        ) = PtyLab.easyInitialize("example:simulation_cpm", operationMode="CPM")

        self.reconstruction = reconstruction
        self.ePIE_engine = ePIE_engine

    def test__move_data_to_cpu(self):
        """
        Move data to CPU even though it's already there. This should not give us an error.
        """
        self.ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
        self.ePIE_engine._move_data_to_cpu()
        self.ePIE_engine._move_data_to_cpu()
        # test that things are actually on the CPU
        assert type(self.ePIE_engine.reconstruction.object) is np.ndarray
        # print(type(self.ePIE_engine.reconstruction.object))

    @unittest.skipIf(cp is None, "no GPU available")
    def test__move_data_to_gpu(self):
        self.ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
        self.ePIE_engine._move_data_to_gpu()
        self.ePIE_engine._move_data_to_gpu()
        assert type(self.ePIE_engine.reconstruction.object) is cp.ndarray


    def test_position_correction(self):
        import time
        rowShifts = np.array([-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        # self.colShifts = dx.flatten()#np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        colShifts = np.array([-2, -1,  0,  1,  2] * 5)
        xp = cp

        Opatch = xp.random.rand(513,513)
        O = xp.roll(Opatch, axis=(-2,-1), shift=(1,-1))

        t0 = time.time()
        for i in range(100):
            cc = xp.zeros((len(rowShifts), 1))
            for shifts in range(len(rowShifts)):
                tempShift = xp.roll(Opatch, rowShifts[shifts], axis=-2)
                # shiftedImages[shifts, ...] = xp.roll(tempShift, self.colShifts[shifts], axis=-1)
                shiftedImages = xp.roll(tempShift, colShifts[shifts], axis=-1)
                cc[shifts] = xp.squeeze(
                    xp.sum(shiftedImages.conj() * O, axis=(-2, -1))
                )
                del tempShift, shiftedImages
            cc = abs(cc)
            cc = cc.reshape(5,5).get()
        t1 = time.time()
        print('CC: ', t1 - t0)

        # new code
        # time it

        t0 = time.time()
        for i in range(100):
            rowShifts, colShifts = xp.mgrid[-2:3, -2:3]
            rowShifts = rowShifts.flatten()
            colShifts = colShifts.flatten()
            print(dy, dx)
            FT_O = xp.fft.fft2(O)
            FT_Op = xp.fft.fft2(Opatch)
            xcor = xp.fft.ifft2(FT_O*FT_Op.conj())
            xcor = abs(xp.fft.fftshift(xcor))
            dy, dx = xp.unravel_index(xp.argmax(xcor), xcor.shape)
            dx = dx.get()
        t1 = time.time()
        print('FT: ', t1-t0)
        N = xcor.shape[-1]
        sy = slice(N//2-len(cc)//2, N//2-len(cc)//2+len(cc))
        print(' Xcor:')
        print(xcor[sy, sy])

        xp.testing.assert_allclose(xcor[sy, sy], cc)


