import logging

import numpy as np
import pytest

import PtyLab
from PtyLab.Params.Params import _check_gpu_availability
from PtyLab.utils.gpuUtils import cp

# Ask the library the same question it asks itself when it sets gpuSwitch, so a
# test can never disagree with the code path it is exercising.
HAS_GPU = bool(_check_gpu_availability())


@pytest.fixture
def engine_setup():
    experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(
        "example:simulation_cpm", operationMode="CPM"
    )
    return reconstruction, ePIE_engine


def test_move_data_to_cpu(engine_setup):
    reconstruction, ePIE_engine = engine_setup
    ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
    ePIE_engine._move_data_to_cpu()
    ePIE_engine._move_data_to_cpu()
    assert type(ePIE_engine.reconstruction.object) is np.ndarray


@pytest.mark.skipif(not HAS_GPU, reason="no GPU available")
def test_move_data_to_gpu(engine_setup):
    reconstruction, ePIE_engine = engine_setup
    ePIE_engine.reconstruction.logger.setLevel(logging.DEBUG)
    ePIE_engine._move_data_to_gpu()
    ePIE_engine._move_data_to_gpu()
    assert type(ePIE_engine.reconstruction.object) is cp.ndarray


@pytest.mark.skip(reason="incomplete/experimental test")
def test_position_correction(engine_setup):
    pass
