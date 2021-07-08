from unittest import TestCase
import logging
logging.basicConfig(level=logging.DEBUG)
import fracPy

class TestBaseEngine(TestCase):
    def setUp(self) -> None:
        experimentalData, reconstruction, params, monitor, ePIE_engine = fracPy.easyInitialize('example:simulation_cpm',
                                                                                               operationMode='CPM')

        self.reconstruction = reconstruction
        self.ePIE_engine = ePIE_engine

    def test__move_data_to_cpu(self):
        """
        Move data to CPU even though it's already there. This should not give us an error.
        """
        self.ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
        self.ePIE_engine._move_data_to_cpu()
        self.ePIE_engine._move_data_to_cpu()
        # test that things are actually on the CPU
        print(type(self.ePIE_engine.reconstruction.object))

    def test__move_data_to_gpu(self):
        self.ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
        self.ePIE_engine._move_data_to_gpu()
        self.ePIE_engine._move_data_to_gpu()
        print(type(self.ePIE_engine.reconstruction.object))




