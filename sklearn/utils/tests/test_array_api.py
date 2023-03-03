import numpy
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
import pytest

from sklearn.base import BaseEstimator
from sklearn.utils._array_api import get_namespace
import array_api_compat

from sklearn.utils._array_api import _asarray_with_order
from sklearn.utils._array_api import _convert_to_numpy
from sklearn.utils._array_api import _estimator_with_converted_arrays

pytestmark = pytest.mark.filterwarnings(
    "ignore:The numpy.array_api submodule:UserWarning"
)


def test_get_namespace_ndarray():
    """Test get_namespace on NumPy ndarrays."""
    X_np = numpy.asarray([[1, 2, 3]])
    xp_out, is_array = get_namespace(X_np)
    assert is_array
    assert xp_out is array_api_compat.numpy


def test_get_namespace_array_api():
    """Test get_namespace for ArrayAPI arrays."""
    xp = pytest.importorskip("numpy.array_api")

    X_xp = xp.asarray([[1, 2, 3]])
    xp_out, is_array = get_namespace(X_xp)
    assert is_array
    assert xp_out.__name__ == "numpy.array_api"

def test_get_namespace_list():
    """Test get_namespace for lists."""

    X = [1, 2, 3]
    xp_out, is_array = get_namespace(X)
    assert not is_array
    assert xp_out is numpy


@pytest.mark.parametrize("is_array_api", [True, False])
def test_asarray_with_order(is_array_api):
    """Test _asarray_with_order passes along order for NumPy arrays."""
    if is_array_api:
        xp = pytest.importorskip("numpy.array_api")
    else:
        xp = numpy

    X = xp.asarray([1.2, 3.4, 5.1])
    X_new = _asarray_with_order(X, order="F")

    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags["F_CONTIGUOUS"]


class _AdjustableNameAPITestWrapper:
    """API wrapper that has an adjustable name. Used for testing."""

    def __init__(self, array_namespace, name):
        self._namespace = array_namespace
        self.__name__ = name

    def __getattr__(self, name):
        return getattr(self._namespace, name)


def test_asarray_with_order_ignored():
    """Test _asarray_with_order ignores order for Generic ArrayAPI."""
    xp = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(xp, "unknown_namespace")

    X = numpy.asarray([[1.2, 3.4, 5.1], [3.4, 5.5, 1.2]], order="C")
    X = xp_.asarray(X)

    X_new = _asarray_with_order(X, order="F", xp=xp_)

    X_new_np = numpy.asarray(X_new)
    assert X_new_np.flags["C_CONTIGUOUS"]
    assert not X_new_np.flags["F_CONTIGUOUS"]


def test_convert_to_numpy_error():
    """Test convert to numpy errors for unsupported namespaces."""
    xp = pytest.importorskip("numpy.array_api")
    xp_ = _AdjustableNameAPITestWrapper(xp, "unknown_namespace")

    X = xp_.asarray([1.2, 3.4])

    msg = "unknown_namespace is an unsupported namespace"
    with pytest.raises(ValueError, match=msg):
        _convert_to_numpy(X, xp=xp_)

@pytest.mark.parametrize("library", ["cupy", "torch", "cupy.array_api"])
def test_convert_to_numpy_gpu(library):
    xp = pytest.importorskip(library)

    if library == "torch":
        if not xp.has_cuda:
            pytest.skip("test requires cuda")
        X_gpu = xp.asarray([1.0, 2.0, 3.0], device="cuda")
    else:
        X_gpu = xp.asarray([1.0, 2.0, 3.0])

    X_cpu = _convert_to_numpy(X_gpu, xp=xp)
    expected_output = numpy.asarray([1.0, 2.0, 3.0])
    assert_allclose(X_cpu, expected_output)


class SimpleEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self.X_ = X
        self.n_features_ = X.shape[0]
        return self


@pytest.mark.parametrize("array_namespace", ["numpy.array_api", "cupy.array_api"])
def test_convert_estimator_to_ndarray(array_namespace):
    """Convert estimator attributes to ndarray."""
    xp = pytest.importorskip(array_namespace)

    if array_namespace == "numpy.array_api":
        converter = lambda array: numpy.asarray(array)  # noqa
    else:  # pragma: no cover
        converter = lambda array: array._array.get()  # noqa

    X = xp.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X)

    new_est = _estimator_with_converted_arrays(est, converter)
    assert isinstance(new_est.X_, numpy.ndarray)


def test_convert_estimator_to_array_api():
    """Convert estimator attributes to ArrayAPI arrays."""
    xp = pytest.importorskip("numpy.array_api")

    X_np = numpy.asarray([[1.3, 4.5]])
    est = SimpleEstimator().fit(X_np)

    new_est = _estimator_with_converted_arrays(est, lambda array: xp.asarray(array))
    assert hasattr(new_est.X_, "__array_namespace__")
