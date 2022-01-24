"""Tools to support array_api."""
import numpy
from .._config import get_config

from contextlib import nullcontext


# There are more clever ways to wrap the API to ignore kwargs, but I am writing them out
# explicitly for demonstration purposes
class _ArrayAPIWrapper:
    def __init__(self, array_namespace):
        self._array_namespace = array_namespace

    def __getattr__(self, name):
        return getattr(self._array_namespace, name)

    def errstate(self, *args, **kwargs):
        # errstate not in `array_api`
        return nullcontext()

    def astype(self, x, dtype, *, copy=True, **kwargs):
        # ignore parameters that is not supported by array-api
        f = self._array_namespace.astype
        return f(x, dtype, copy=copy)

    def asarray(self, obj, dtype=None, device=None, copy=True, **kwargs):
        f = self._array_namespace.asarray
        return f(obj, dtype=dtype, device=device, copy=copy)

    def may_share_memory(self, *args, **kwargs):
        # The safe choice is to return True all the time
        return True

    def asanyarray(self, array, *args, **kwargs):
        # noop
        return array

    def concatenate(self, arrays, *, axis=0, **kwargs):
        # ignore parameters that is not supported by array-api
        f = self._array_namespace.concat
        return f(arrays, axis=axis)

    def unique(self, x, return_inverse=False):
        if return_inverse:
            f = self._array_namespace.unique_inverse
        else:
            f = self._array_namespace.unique_values
        return f(x)

    def bincount(self, x):
        f = self._array_namespace.unique_counts
        return f(x)[1]

    @property
    def VisibleDeprecationWarning(self):
        return DeprecationWarning


class _NumPyApiWrapper:
    def __getattr__(self, name):
        return getattr(numpy, name)

    def astype(self, x, dtype, *args, **kwargs):
        # astype is not defined in the top level numpy namespace
        return x.astype(dtype, *args, **kwargs)


def get_namespace(*xs):
    # `xs` contains one or more arrays, or possibly Python scalars (accepting
    # those is a matter of taste, but doesn't seem unreasonable).
    # Returns a tuple: (array_namespace, is_array_api)

    if not get_config()["array_api_dispatch"]:
        return _NumPyApiWrapper(), False

    namespaces = {
        x.__array_namespace__() if hasattr(x, "__array_namespace__") else None
        for x in xs
        if not isinstance(x, (bool, int, float, complex))
    }

    if not namespaces:
        # one could special-case np.ndarray above or use np.asarray here if
        # older numpy versions need to be supported.
        raise ValueError("Unrecognized array input")

    if len(namespaces) != 1:
        raise ValueError(f"Multiple namespaces for array inputs: {namespaces}")

    (xp,) = namespaces
    if xp is None:
        # Use numpy as default
        return _NumPyApiWrapper(), False

    return _ArrayAPIWrapper(xp), True
