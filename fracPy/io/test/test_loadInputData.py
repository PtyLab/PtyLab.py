from unittest import TestCase
from fracPy.io import get_example_data_folder
from fracPy.io.readHdf5 import loadInputData
import logging


class TestLoadInputData(TestCase):
    def test_loadInputData(self):
        """
        Load the input data. We know that the shape of the input should be 441,128,128,
        so check that it's loaded correctly.

        # TODO: The filename of example_data should change, and hence this function has to be reimplemented too.
        :return:
        """
        # first, get the filename of the first .hdf5 dataset
        example_data_folder = get_example_data_folder()
        filename = example_data_folder / 'usaft2_441_LED/Image_section_0_color_1.hdf5'
        result = loadInputData(filename)
        self.assertEqual(result['ptychogram'].shape, (441,128,128))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import unittest
    unittest.main()