"""Tools to support array_api."""
import numpy
import scipy.special as special

# XXX: Direct input for testing, array_api_compat should be vendored
import array_api_compat
import array_api_compat.numpy
from array_api_compat import device, size  # noqa

from .._config import get_config


def _is_numpy_namespace(xp):
    return xp.__name__ in {"numpy", "array_api_compat.numpy", "numpy.array_api"}


class _ArrayAPIWrapper:
    """sklearn specific Array API compatibility wrapper

    This wrapper makes it possible for scikit-learn maintainers to
    deal with discrepancies between different implementations of the
    Python array API standard and its evolution over time.

    The Python array API standard specification:
    https://data-apis.org/array-api/latest/

    Documentation of the NumPy implementation:
    https://numpy.org/neps/nep-0047-array-api-standard.html
    """

    def __init__(self, array_namespace):
        self._namespace = array_namespace

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def take(self, X, indices, *, axis=0):
        # When array_api supports `take` we can use this directly
        # https://github.com/data-apis/array-api/issues/177
        if self._namespace.__name__ == "numpy.array_api":
            X_np = numpy.take(X, indices, axis=axis)
            return self._namespace.asarray(X_np)

        # We only support axis in (0, 1) and ndim in (1, 2) because that is all we need
        # in scikit-learn
        if axis not in {0, 1}:
            raise ValueError(f"Only axis in (0, 1) is supported. Got {axis}")

        if X.ndim not in {1, 2}:
            raise ValueError(f"Only X.ndim in (1, 2) is supported. Got {X.ndim}")

        if axis == 0:
            if X.ndim == 1:
                selected = [X[i] for i in indices]
            else:  # X.ndim == 2
                selected = [X[i, :] for i in indices]
        else:  # axis == 1
            selected = [X[:, i] for i in indices]
        return self._namespace.stack(selected, axis=axis)

    def isdtype(self, dtype, kind):
        """Returns a boolean indicating whether a provided dtype is of type "kind".

        Included in the v2022.12 of the Array API spec.
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
        """
        if isinstance(kind, tuple):
            return any(self._isdtype_single(dtype, k) for k in kind)
        else:
            return self._isdtype_single(dtype, kind)

    def _isdtype_single(self, dtype, kind):
        xp = self._namespace
        if isinstance(kind, str):
            if kind == "bool":
                return dtype == xp.bool
            elif kind == "signed integer":
                return dtype in {xp.int8, xp.int16, xp.int32, xp.int64}
            elif kind == "unsigned integer":
                return dtype in {xp.uint8, xp.uint16, xp.uint32, xp.uint64}
            elif kind == "integral":
                return self.isdtype(dtype, ("signed integer", "unsigned integer"))
            elif kind == "real floating":
                return {xp.float32, xp.float64}
            elif kind == "complex floating":
                return {xp.complex64, xp.complex128}
            elif kind == "numeric":
                return self.isdtype(
                    dtype, ("integral", "real floating", "complex floating")
                )
            else:
                raise ValueError(f"Unrecognized data type kind: {kind!r}")
        else:
            return dtype == kind


def get_namespace(*arrays):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API
    compatible namespace object, if any. NumPy 1.22 and later can
    construct such containers using the `numpy.array_api` namespace
    for instance.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, an instance of the
    `_NumPyApiWrapper` compatibility wrapper is returned instead.

    Namespace support is not enabled by default. To enabled it
    call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise an instance of the `_NumPyApiWrapper`
    compatibility wrapper is always returned irrespective of
    the fact that arrays implement the `__array_namespace__`
    protocol or not.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    Returns
    -------
    namespace : module
        Namespace shared by array objects. If any of the `arrays` are not arrays,
        the namespace defaults to NumPy.

    is_array : bool
        True if arrays are recognized arrays.
    """
    return _get_namespace(
        *arrays, array_api_dispatch=get_config()["array_api_dispatch"]
    )


def _get_namespace(*arrays, array_api_dispatch=False):
    if not array_api_dispatch:
        return array_api_compat.numpy, False
    try:
        namespace, is_array = array_api_compat.get_namespace(*arrays), True
    except TypeError as e:
        if str(e).startswith("The input is not a supported array type"):
            return array_api_compat.numpy, False
        raise

    if namespace.__name__ in {"numpy.array_api", "cupy.array_api"}:
        namespace = _ArrayAPIWrapper(namespace)

    return namespace, is_array


def _expit(X):
    xp, _ = get_namespace(X)
    if _is_numpy_namespace(xp):
        return xp.asarray(special.expit(numpy.asarray(X)))

    return 1.0 / (1.0 + xp.exp(-X))


def _asarray_with_order(array, dtype=None, order=None, copy=None, xp=None):
    """Helper to support the order kwarg only for NumPy-backed arrays

    Memory layout parameter `order` is not exposed in the Array API standard,
    however some input validation code in scikit-learn needs to work both
    for classes and functions that will leverage Array API only operations
    and for code that inherently relies on NumPy backed data containers with
    specific memory layout constraints (e.g. our own Cython code). The
    purpose of this helper is to make it possible to share code for data
    container validation without memory copies for both downstream use cases:
    the `order` parameter is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.
    """
    if xp is None:
        xp, _ = get_namespace(array)
    if _is_numpy_namespace(xp):
        # Use NumPy API to support order
        if copy is True:
            array = numpy.array(array, order=order, dtype=dtype)
        else:
            array = numpy.asarray(array, order=order, dtype=dtype)
        return xp.asarray(array)
    else:
        return xp.asarray(array, dtype=dtype, copy=copy)


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__

    if _is_numpy_namespace(xp):
        return numpy.asarray(array)
    if xp_name == "cupy.array_api":
        return array._array.get()
    elif xp_name in {"array_api_compat.cupy", "cupy"}:
        return array.get()
    elif xp_name in {"array_api_compat.torch", "torch"}:
        return array.cpu().numpy()
    else:
        raise ValueError(f"{xp_name} is an unsupported namespace")


def _estimator_with_converted_arrays(estimator, converter):
    """Create new estimator which converting all attributes that are arrays.

    Parameters
    ----------
    estimator : Estimator
        Estimator to convert

    converter : callable
        Callable that takes an array attribute and returns the converted array.

    Returns
    -------
    new_estimator : Estimator
        Convert estimator
    """
    from sklearn.base import clone

    new_estimator = clone(estimator)
    for key, attribute in vars(estimator).items():
        _, is_array = _get_namespace(attribute, array_api_dispatch=True)
        if is_array:
            attribute = converter(attribute)
        setattr(new_estimator, key, attribute)
    return new_estimator
