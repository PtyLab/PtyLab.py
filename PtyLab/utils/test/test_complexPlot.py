import matplotlib

matplotlib.use("tkagg")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from PtyLab.utils.visualisation import complex2rgb, complexPlot


class TestComplexPlot(unittest.TestCase):
    def test_complex_plot(self):
        testComplexArray = np.ones((100, 100)) * (1 + 1j)
        testRGBArray = complex2rgb(testComplexArray)
        complexPlot(testRGBArray)
        plt.show()


if __name__ == "__main__":
    unittest.main()
