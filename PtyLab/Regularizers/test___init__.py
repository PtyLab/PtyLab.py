import time
from unittest import TestCase
import numpy as np
from numpy.testing import assert_allclose

from PtyLab.Regularizers import divergence, grad_TV


class Test(TestCase):
    def setUp(self) -> None:
        self.nlambda = 1
        self.nosm = 1
        self.nslice = 1
        self.No = 380
        self.shape_O = (
            self.nlambda,
            self.nosm,
            1,
            self.nslice,
            self.No,
            self.No
        )
        self.object = np.random.rand(*self.shape_O) + 1j * np.random.rand(*self.shape_O)
        self.object -= 0.5 + 0.5j

    def test_divergence(self):

        # copied from wilhelm
        t0 = time.time()
        for i in range(10):
            epsilon = 1e-2
            gradient = np.gradient(self.object, axis=(4, 5))
            norm = (gradient[0]+gradient[1])**2
            temp = [gradient[0] / np.sqrt(norm+epsilon), gradient[1] / np.sqrt(norm+epsilon)]
            TV_update = divergence(temp)
        t1= time.time()
        print(t1 - t0)

        # new method, should be faster and more consuming
        t0 = time.time()
        for i in range(10):
            TV_update_2 = grad_TV(self.object, epsilon)
        t1 = time.time()
        print(t1 - t0)
        assert_allclose(TV_update, TV_update_2)

