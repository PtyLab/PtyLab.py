import timeit
from unittest import TestCase
from fracPy import easyInitialize
import logging
logging.basicConfig()

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
        reconstruction.auto_scale_object_size = True
        print(reconstruction.positions[-1])
        reconstruction.zo = reconstruction.zo / 6
        with self.assertRaises(ValueError):
            # Make sure it raises a loud error
            reconstruction.positions[-1]

        reconstruction.pad_or_shrink_object()

        # check that it is performed fast so we can run it after every loop

        res = timeit.timeit("reconstruction.pad_or_shrink_object()",globals={'reconstruction': reconstruction},
                            number=1000)
        print(f'Speed: {1000/res} pad-or-shrinks per second')
        reconstruction.positions[-1]
