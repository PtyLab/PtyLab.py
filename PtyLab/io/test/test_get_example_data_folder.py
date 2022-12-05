from unittest import TestCase
from PtyLab.io import getExampleDataFolder


class TestGet_example_data_folder(TestCase):
    def test_example_folder_exists(self):
        """
        Test that the path returned by getExampleDataFolder exists.
        :return:
        """
        example_data_folder = getExampleDataFolder()
        self.assertTrue(
            example_data_folder.exists(),
            "example data folder returned does not exist on the filesystem",
        )

    def test_simulationTiny_in_example_data(self):
        """
        The test folder always has simulationTiny.hdf5 in it. Check that it's present.
        :return:
        """
        example_data_folder = getExampleDataFolder()
        matlabfile = example_data_folder / "simulationTiny.hdf5"
        self.assertTrue(
            matlabfile.exists(),
            "`simulationTiny.hdf5` is not present in the example data folder",
        )
