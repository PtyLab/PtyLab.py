from unittest import TestCase
from fracPy.io import exampleLoader
import numpy as np


class TestLoad_example(TestCase):
    def test_load_example(self):
        """
        Check that load_example works. Based on the input we know that it should give us a dataset
        with 32 pictures in it
        :return:
        """
        archive = exampleLoader.loadExample('simulationTiny.mat')
        # check that we can read something
        self.assertEqual(np.array(archive['Nd'], int), 64)
