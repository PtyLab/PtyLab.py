import matplotlib

matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import unittest
from PtyLab.Monitor.TensorboardMonitor import center_angle
from scipy import ndimage
import numpy as np


class AngleStuff(unittest.TestCase):
    def test_center_angle(self):
        N = 128
        Ein = np.fft.fftshift(ndimage.fourier_shift(np.ones((N, N)), [5, 0]))

        Ein_c = center_angle(Ein)

        # print('hoihoi')
        plt.subplot(121)
        plt.imshow(np.angle(Ein), cmap="hsv", clim=[-np.pi, np.pi])
        plt.subplot(122)
        # print(Ein_c)
        plt.imshow(np.angle(Ein_c), cmap="hsv", clim=[-np.pi, np.pi])
        plt.show()


if __name__ == "__main__":
    unittest.main()
