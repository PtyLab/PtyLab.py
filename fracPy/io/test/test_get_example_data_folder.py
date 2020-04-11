from unittest import TestCase
from fracPy.io import get_example_data_folder

class TestGet_example_data_folder(TestCase):
    def test_example_folder_exists(self):
        """
        Test that the path returned by get_example_data_folder exists.
        :return:
        """
        example_data_folder = get_example_data_folder()
        self.assertTrue(example_data_folder.exists(), 'example data folder returned does not exist on the filesystem')

    def test_simulationTiny_in_example_data(self):
        """
        The test folder always has simulationTiny.mat in it. Check that it's present.
        :return:
        """
        example_data_folder = get_example_data_folder()
        matlabfile = example_data_folder / 'simulationTiny.mat'
        self.assertTrue(matlabfile.exists(), '`simulationTiny.mat` is not present in the example data folder')

