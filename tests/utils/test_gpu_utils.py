"""Tests for the host/device transfer helpers in PtyLab.utils.gpuUtils.

These helpers decide, for every array the reconstruction owns, which device it
lives on and what precision it is stored in. Both decisions are invisible at the
call site -- ``_move_data_to_gpu()`` takes no arguments and says nothing about
dtype -- so they are pinned here rather than left to be discovered by a
reconstruction that silently loses precision.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from PtyLab.utils.gpuUtils import (
    CP_AVAILABLE,
    asCupyArray,
    asNumpyArray,
    cp,
    getArrayModule,
    isGpuArray,
    transfer_fields_to_cpu,
    transfer_fields_to_gpu,
)


def _gpu_available() -> bool:
    """cupy importable *and* a device actually usable.

    Taken through the module under test, so these tests see the same cupy the
    library does rather than importing a second view of it.
    """
    if not CP_AVAILABLE:
        return False
    try:
        return bool(cp.cuda.is_available())
    except RuntimeError:  # driver present but unusable
        return False


HAS_GPU = _gpu_available()

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="no GPU available")

# Stand-in for Reconstruction/ExperimentalData: an object carrying named fields
# that the transfer helpers reach by getattr/setattr.
Holder = SimpleNamespace


class ReadOnlyField:
    """An object whose field cannot be assigned, as a read-only property."""

    def __init__(self, value):
        self._value = value

    @property
    def field(self):
        return self._value


@pytest.fixture
def logger():
    return logging.getLogger("test_gpu_utils")


def test_getArrayModule_and_isGpuArray_on_the_host():
    host = np.zeros(3)

    assert getArrayModule(host) is np
    assert isGpuArray(host) is False


@requires_gpu
def test_getArrayModule_and_isGpuArray_on_the_device():
    device = cp.zeros(3)

    assert getArrayModule(device) is cp
    assert isGpuArray(device) is True


@requires_gpu
def test_asNumpyArray_brings_an_array_back_to_the_host():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = asNumpyArray(cp.asarray(values))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, values)


def test_asNumpyArray_passes_a_host_array_through():
    values = np.arange(3)

    assert asNumpyArray(values) is values


# ---------------------------------------------------------------------------
# dtype policy
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize(
    "source, expected",
    [
        (np.float64, np.float32),
        (np.float32, np.float32),
        (np.int32, np.float32),
        (np.complex128, np.complex64),
        (np.complex64, np.complex64),
    ],
)
def test_asCupyArray_downcasts_to_single_precision(source, expected):
    """dtype='auto' halves the precision of everything it moves.

    This is deliberate -- single precision is what the engines run in -- but it
    happens silently on every transfer, so a float64 array handed to the GPU
    comes back float32. Pinned so the choice cannot drift unnoticed.
    """
    result = asCupyArray(np.zeros(3, dtype=source))

    assert result.dtype == expected


@requires_gpu
def test_asCupyArray_honours_an_explicit_dtype():
    result = asCupyArray(np.zeros(3, dtype=np.float32), dtype=np.float64)

    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# field transfer
# ---------------------------------------------------------------------------


@requires_gpu
def test_transfer_fields_moves_only_the_named_fields(logger):
    holder = Holder(moved=np.zeros(4, dtype=np.float32), left=np.zeros(4))

    transfer_fields_to_gpu(holder, ["moved"], logger)

    assert isGpuArray(holder.moved)
    assert not isGpuArray(holder.left)


@requires_gpu
def test_transfer_fields_round_trips(logger):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    holder = Holder(field=values.copy())

    transfer_fields_to_gpu(holder, ["field"], logger)
    assert isGpuArray(holder.field)

    transfer_fields_to_cpu(holder, ["field"], logger)
    assert isinstance(holder.field, np.ndarray)
    np.testing.assert_array_equal(holder.field, values)


@requires_gpu
def test_transfer_fields_to_gpu_applies_its_dtype_argument(logger):
    """The dtype argument has to reach asCupyArray.

    It was accepted and documented but silently dropped, so a caller asking for
    double precision quietly got single. Nothing in the library passes it today,
    which is exactly why this needs a test rather than a caller.
    """
    holder = Holder(field=np.zeros(3, dtype=np.float64))

    transfer_fields_to_gpu(holder, ["field"], logger, dtype=np.float64)

    assert holder.field.dtype == np.float64

    holder = Holder(field=np.zeros(3, dtype=np.float64))
    transfer_fields_to_gpu(holder, ["field"], logger)  # dtype='auto'

    assert holder.field.dtype == np.float32


def test_transfer_fields_skips_fields_that_are_not_defined(logger):
    """Both transfer functions are called with a superset of the fields.

    possible_GPU_fields lists names that only some engines define, so skipping
    an absent field is the normal case, not an error.
    """
    holder = Holder(present=np.zeros(3, dtype=np.float32))

    transfer_fields_to_cpu(holder, ["present", "absent"], logger)

    assert not hasattr(holder, "absent")


@requires_gpu
def test_transfer_fields_to_gpu_reports_a_field_it_cannot_set(logger, caplog):
    """A read-only property must fail loudly, naming the field.

    Reconstruction exposes several derived attributes as read-only properties;
    listing one for transfer by mistake would otherwise raise from inside
    setattr with no indication of which name was at fault.
    """
    holder = ReadOnlyField(np.zeros(3, dtype=np.float32))

    with caplog.at_level(logging.ERROR), pytest.raises(AttributeError):
        transfer_fields_to_gpu(holder, ["field"], logger)

    assert "field" in caplog.text
