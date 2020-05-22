import matplotlib
matplotlib.use('tkagg')
import unittest
import numpy as np
import matplotlib.pyplot as plt
from fracPy.utils.visualisation import complex_to_rgb, complex_plot



class TestComplexPlot(unittest.TestCase):
    def test_complex_plot(self):
        testComplexArray = np.ones((100,100))*(1+1j)
        testRGBArray = complex_to_rgb(testComplexArray)
        complex_plot(testRGBArray)
        plt.show()


if __name__ == '__main__':
    unittest.main()
