from unittest import TestCase
from PtyLab.io import readExample
import numpy as np
import logging

logging.basicConfig()


class TestRead_example(TestCase):
    def test_read_example(self):
        """
        Check that loadExample works. Based on the input we know that it should give us a dataset
        with 32 pictures in it
        :return:
        """

        readExample.listExamples()
        archive = readExample.loadExample("simulation_fpm")
        # check that we can read something
        self.assertEqual(256, np.array(archive["Nd"], int))

        archive = readExample.loadExample("simulationTiny")
        self.assertEqual(64, np.array(archive["Nd"], int))
