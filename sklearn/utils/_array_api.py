"""Tools to support array_api."""
import numpy
from scipy.special import logsumexp as sp_logsumexp
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

    def astype(self, x, dtype, copy=True, **kwargs):
        # ignore parameters that is not supported by array-api
        f = self._array_namespace.astype
        return f(x, dtype, copy=copy)

    def asarray(self, obj, dtype=None, device=None, copy=None, **kwargs):
        f = self._array_namespace.asarray
        return f(obj, dtype=dtype, device=device, copy=copy)

    def array(self, obj, dtype=None, device=None, copy=True, **kwargs):
        f = self._array_namespace.asarray
        return f(obj, dtype=dtype, device=device, copy=copy)

    def asanyarray(self, obj, *args, **kwargs):
        # no-op for now
        return obj

    def may_share_memory(self, *args, **kwargs):
        # The safe choice is to return True all the time
        return True


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


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    np, is_array_api = get_namespace(a)

    # Use SciPy if a is an ndarray
    if not is_array_api:
        return sp_logsumexp(
            a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign
        )

    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.0  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.max(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    s = np.sum(tmp, axis=axis, keepdims=keepdims)
    if return_sign:
        sgn = np.sign(s)
        s *= sgn  # /= makes more sense but we need zero -> zero
    out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out
