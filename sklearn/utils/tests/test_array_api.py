import numpy
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from sklearn.base import BaseEstimator
from sklearn.utils._array_api import get_namespace
from sklearn.utils._array_api import _NumPyApiWrapper
from sklearn.utils._array_api import _ArrayAPIWrapper
from sklearn.utils._array_api import _asarray_with_order
from sklearn.utils._array_api import _convert_to_numpy
from sklearn.utils._array_api import _estimator_with_converted_arrays
from sklearn.utils._testing import skip_if_array_api_compat_not_configured

from sklearn._config import config_context

pytestmark = pytest.mark.filterwarnings(
    "ignore:The numpy.array_api submodule:UserWarning"
)


@pytest.mark.parametrize("X", [numpy.asarray([1, 2, 3]), [1, 2, 3]])
def test_get_namespace_ndarray_default(X):
    """Check that get_namespace returns NumPy wrapper"""
    xp_out, is_array_api_compliant = get_namespace(X)
    assert isinstance(xp_out, _NumPyApiWrapper)
    assert not is_array_api_compliant


def test_get_namespace_ndarray_creation_device():
    """Check expected behavior with device and creation functions."""
    X = numpy.asarray([1, 2, 3])
    xp_out, _ = get_namespace(X)

    full_array = xp_out.full(10, fill_value=2.0, device="cpu")
    assert_allclose(full_array, [2.0] * 10)

    with pytest.raises(ValueError, match="Unsupported device"):
        xp_out.zeros(10, device="cuda")


@skip_if_array_api_compat_not_configured
def test_get_namespace_ndarray_with_dispatch():
    """Test get_namespace on NumPy ndarrays."""
    array_api_compat = pytest.importorskip("array_api_compat")

    X_np = numpy.asarray([[1, 2, 3]])

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X_np)
        assert is_array_api_compliant
        assert xp_out is array_api_compat.numpy


@skip_if_array_api_compat_not_configured
def test_get_namespace_array_api():
    """Test get_namespace for ArrayAPI arrays."""
    xp = pytest.importorskip("numpy.array_api")

    X_np = numpy.asarray([[1, 2, 3]])
    X_xp = xp.asarray(X_np)
    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X_xp)
        assert is_array_api_compliant
        assert isinstance(xp_out, _ArrayAPIWrapper)

        xp_out, is_array_api_compliant = get_namespace(1)
        assert isinstance(xp_out, _NumPyApiWrapper)
        assert not is_array_api_compliant


class _AdjustableNameAPITestWrapper(_ArrayAPIWrapper):
    """API wrapper that has an adjustable name. Used for testing."""

    def __init__(self, array_namespace, name):
        super().__init__(array_namespace=array_namespace)
        self.__name__ = name


def test_array_api_wrapper_astype():
    """Test _ArrayAPIWrapper for ArrayAPIs that is not NumPy."""
    numpy_array_api = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, "wrapped_numpy.array_api")
    xp = _ArrayAPIWrapper(xp_)

    X = xp.asarray(([[1, 2, 3], [3, 4, 5]]), dtype=xp.float64)
    X_converted = xp.astype(X, xp.float32)
    assert X_converted.dtype == xp.float32

    X_converted = xp.asarray(X, dtype=xp.float32)
    assert X_converted.dtype == xp.float32


def test_array_api_wrapper_take_for_numpy_api():
    """Test that fast path is called for numpy.array_api."""
    numpy_array_api = pytest.importorskip("numpy.array_api")
    # USe the same name as numpy.array_api
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, "numpy.array_api")
    xp = _ArrayAPIWrapper(xp_)

    X = xp.asarray(([[1, 2, 3], [3, 4, 5]]), dtype=xp.float64)
    X_take = xp.take(X, xp.asarray([1]), axis=0)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X, [1], axis=0))


def test_array_api_wrapper_take():
    """Test _ArrayAPIWrapper API for take."""
    numpy_array_api = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(numpy_array_api, "wrapped_numpy.array_api")
    xp = _ArrayAPIWrapper(xp_)

    # Check take compared to NumPy's with axis=0
    X_1d = xp.asarray([1, 2, 3], dtype=xp.float64)
    X_take = xp.take(X_1d, xp.asarray([1]), axis=0)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X_1d, [1], axis=0))

    X = xp.asarray(([[1, 2, 3], [3, 4, 5]]), dtype=xp.float64)
    X_take = xp.take(X, xp.asarray([0]), axis=0)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X, [0], axis=0))

    # Check take compared to NumPy's with axis=1
    X_take = xp.take(X, xp.asarray([0, 2]), axis=1)
    assert hasattr(X_take, "__array_namespace__")
    assert_array_equal(X_take, numpy.take(X, [0, 2], axis=1))

    with pytest.raises(ValueError, match=r"Only axis in \(0, 1\) is supported"):
        xp.take(X, xp.asarray([0]), axis=2)

    with pytest.raises(ValueError, match=r"Only X.ndim in \(1, 2\) is supported"):
        xp.take(xp.asarray([[[0]]]), xp.asarray([0]), axis=0)


@pytest.mark.parametrize(
    "is_array_api_compliant",
    [pytest.param(True, marks=skip_if_array_api_compat_not_configured), False],
)
def test_asarray_with_order(is_array_api_compliant):
    """Test _asarray_with_order passes along order for NumPy arrays."""
    if is_array_api_compliant:
        xp = pytest.importorskip("numpy.array_api")
    else:
        xp = numpy

    X = xp.asarray([1.2, 3.4, 5.1])
    X_new = _asarray_with_order(X, order="F", xp=xp)

    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags["F_CONTIGUOUS"]


def test_asarray_with_order_ignored():
    """Test _asarray_with_order ignores order for Generic ArrayAPI."""
    xp = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(xp, "wrapped.array_api")

    X = numpy.asarray([[1.2, 3.4, 5.1], [3.4, 5.5, 1.2]], order="C")
    X = xp_.asarray(X)

    X_new = _asarray_with_order(X, order="F", xp=xp_)

    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags["C_CONTIGUOUS"]
    assert not X_new_np.flags["F_CONTIGUOUS"]


@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize("library", ["cupy", "torch", "cupy.array_api"])
def test_convert_to_numpy_gpu(library):
    """Check convert_to_numpy for GPU backed libraries."""
    xp = pytest.importorskip(library)

    if library == "torch":
        if not xp.has_cuda:
            pytest.skip("test requires cuda")
        X_gpu = xp.asarray([1.0, 2.0, 3.0], device="cuda")
    else:
        X_gpu = xp.asarray([1.0, 2.0, 3.0])

    X_cpu = _convert_to_numpy(X_gpu)
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    assert_allclose(X_cpu, expected_output)


def test_convert_to_numpy_cpu():
    """Check convert_to_numpy for PyTorch CPU arrays."""
    torch = pytest.importorskip("torch")
    X_torch = torch.asarray([1.0, 2.0, 3.0], device="cpu")

    X_cpu = _convert_to_numpy(X_torch)
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    assert_allclose(X_cpu, expected_output)


class SimpleEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.X_ = X
        self.n_features_ = X.shape[0]
        return self


@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize(
    "array_namespace, converter",
    [
        ("torch", lambda array: array.cpu().numpy()),
        ("numpy.array_api", lambda array: numpy.asarray(array)),
        ("cupy.array_api", lambda array: array._array.get()),
    ],
)
def test_convert_estimator_to_ndarray(array_namespace, converter):
    """Convert estimator attributes to ndarray."""
    xp = pytest.importorskip(array_namespace)

    X = xp.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X)

    new_est = _estimator_with_converted_arrays(est, converter)
    assert isinstance(new_est.X_, numpy.ndarray)


@skip_if_array_api_compat_not_configured
def test_convert_estimator_to_array_api():
    """Convert estimator attributes to ArrayAPI arrays."""
    xp = pytest.importorskip("numpy.array_api")

    X_np = numpy.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X_np)

    new_est = _estimator_with_converted_arrays(est, lambda array: xp.asarray(array))
    assert hasattr(new_est.X_, "__array_namespace__")


@pytest.mark.parametrize("wrapper", [_ArrayAPIWrapper, _NumPyApiWrapper])
def test_get_namespace_array_api_isdtype(wrapper):
    """Test isdtype implementation from _ArrayAPIWrapper and _NumPyApiWrapper."""

    if wrapper == _ArrayAPIWrapper:
        xp_ = pytest.importorskip("numpy.array_api")
        xp = _ArrayAPIWrapper(xp_)
    else:
        xp = _NumPyApiWrapper()

    assert xp.isdtype(xp.float32, "real floating")
    assert xp.isdtype(xp.float64, "real floating")
    assert not xp.isdtype(xp.int32, "real floating")

    assert xp.isdtype(xp.bool, "bool")
    assert not xp.isdtype(xp.float32, "bool")

    assert xp.isdtype(xp.int16, "signed integer")
    assert not xp.isdtype(xp.uint32, "signed integer")

    assert xp.isdtype(xp.uint16, "unsigned integer")
    assert not xp.isdtype(xp.int64, "unsigned integer")

    assert xp.isdtype(xp.int64, "numeric")
    assert xp.isdtype(xp.float32, "numeric")
    assert xp.isdtype(xp.uint32, "numeric")

    assert not xp.isdtype(xp.float32, "complex floating")

    if wrapper == _NumPyApiWrapper:
        assert not xp.isdtype(xp.int8, "complex floating")
        assert xp.isdtype(xp.complex64, "complex floating")
        assert xp.isdtype(xp.complex128, "complex floating")


@pytest.mark.parametrize(
    "array_api_dispatch",
    [pytest.param(True, marks=skip_if_array_api_compat_not_configured), False],
)
def test_get_namespace_list(array_api_dispatch):
    """Test get_namespace for lists."""

    X = [1, 2, 3]
    with config_context(array_api_dispatch=array_api_dispatch):
        xp_out, is_array = get_namespace(X)
        assert not is_array
        assert isinstance(xp_out, _NumPyApiWrapper)


def test_error_when_reshape_is_not_a_tuple():
    """Raise error in reshape when shape is not a tuple."""
    X = numpy.asarray([[1, 2, 3], [3, 4, 5]])
    xp, _ = get_namespace(X)

    with pytest.raises(TypeError, match="shape must be a tuple"):
        xp.reshape(X, -1)
