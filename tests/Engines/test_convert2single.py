import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import PtyLab
from PtyLab import Engines


@pytest.fixture
def engine(generate_simu_hdf5):
    """Initialize a fresh engine with double-precision arrays for each test."""
    experimental_data, reconstruction, _, _, engine = PtyLab.easyInitialize(
        "example:simulation_cpm",
        engine=Engines.ePIE,
        operationMode="CPM",
        dummyMonitor=True,
    )
    reconstruction.probe = reconstruction.probe.astype(np.complex128)
    reconstruction.object = reconstruction.object.astype(np.complex128)
    experimental_data.ptychogram = experimental_data.ptychogram.astype(np.float64)
    engine.test_complex_array = np.full((4, 4), 1.25 + 2.5j, dtype=np.complex128)
    engine.test_real_array = np.full((4, 4), 1.25, dtype=np.float64)
    return engine


@pytest.mark.parametrize(
    "container_name,attribute,expected_dtype",
    [
        ("reconstruction", "probe", np.complex64),
        ("reconstruction", "object", np.complex64),
        ("experimentalData", "ptychogram", np.float32),
        (None, "test_complex_array", np.complex64),
        (None, "test_real_array", np.float32),
    ],
)
def test_convert2single_converts_arrays(engine, container_name, attribute, expected_dtype):
    container = engine if container_name is None else getattr(engine, container_name)
    original = getattr(container, attribute).copy()

    engine.convert2single()

    converted = getattr(container, attribute)
    assert converted.dtype == expected_dtype
    assert converted.shape == original.shape
    assert_allclose(converted, original, rtol=1e-6, atol=1e-7)


def test_convert2single_preserves_other_attributes(engine):
    integers = np.arange(10, dtype=np.int64)
    booleans = np.array([True, False], dtype=np.bool_)
    engine.test_integer_array = integers.copy()
    engine.test_boolean_array = booleans.copy()
    engine.test_string = "PtyLab"
    engine.test_boolean = True

    engine.convert2single()

    assert engine.test_integer_array.dtype == np.int64
    assert_array_equal(engine.test_integer_array, integers)
    assert engine.test_boolean_array.dtype == np.bool_
    assert_array_equal(engine.test_boolean_array, booleans)
    assert engine.test_string == "PtyLab"
    assert engine.test_boolean is True


def test_convert2single_sets_target_dtypes(engine):
    engine.convert2single()

    assert engine.dtype_complex == np.complex64
    assert engine.dtype_real == np.float32
