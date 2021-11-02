from unittest import TestCase
from fracPy import easyInitialize

class TestReconstruction(TestCase):
    def test_positions(self):
        """
         Make sure that when positions are loaded, and the positions
        end up with negative indices or indices that are larger than the number of pixels in the object,
        we get a loud error.

        """
        filePath='example:simulation_cpm'
        experimentalData, reconstruction, params, monitor, ePIE_engine = easyInitialize(filePath,
                                                                                               operationMode='CPM')
        print(reconstruction.positions[-1])
        reconstruction.zo = reconstruction.zo / 6
        with self.assertRaises(ValueError):
            # Make sure it raises a loud error
            reconstruction.positions[-1]
