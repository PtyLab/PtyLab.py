import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData


@pytest.fixture
def data():
    """A small dataset with distinct frames and positions, without file I/O."""
    data = ExperimentalData("test:nodata", operationMode="CPM")
    data.ptychogram = np.arange(1, 20 * 16 * 16 + 1, dtype=np.float32).reshape(
        20, 16, 16
    )
    data.encoder = np.arange(40, dtype=np.float64).reshape(20, 2) * 1e-6
    data._setData()
    return data


@pytest.mark.parametrize("start,end", [(0, 10), (3, 10)])
def test_reduce_positions_updates_data_and_metadata(data, start, end):
    expected_ptychogram = data.ptychogram[start:end].copy()
    expected_encoder = data.encoder[start:end].copy()
    expected_energy = expected_ptychogram.sum(axis=(-1, -2))

    data.reduce_positions(start, end)

    assert_array_equal(data.ptychogram, expected_ptychogram)
    assert_array_equal(data.encoder, expected_encoder)
    assert data.numFrames == end - start
    assert data.energyAtPos.shape == (end - start,)
    assert_allclose(data.energyAtPos, expected_energy)
    assert_allclose(data.maxProbePower, np.sqrt(expected_energy.max()))


def test_set_data_preserves_reduced_positions(data):
    data.reduce_positions(0, 10)
    expected_ptychogram = data.ptychogram.copy()
    expected_encoder = data.encoder.copy()
    expected_energy = data.energyAtPos.copy()

    data._setData()

    assert_array_equal(data.ptychogram, expected_ptychogram)
    assert_array_equal(data.encoder, expected_encoder)
    assert data.numFrames == 10
    assert data.energyAtPos.shape == (10,)
    assert_allclose(data.energyAtPos, expected_energy)
