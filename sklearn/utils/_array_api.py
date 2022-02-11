"""Tools to support array_api."""
import numpy
import scipy.special
from .._config import get_config

# There are more clever ways to wrap the API to ignore kwargs, but I am writing them out
# explicitly for demonstration purposes
class _ArrayAPIWrapper:
    def __init__(self, array_namespace):
        self._namespace = array_namespace

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def astype(self, x, dtype, *, copy=True, casting="unsafe"):
        # support casting for NumPy
        if self._namespace.__name__ == "numpy.aray_api":
            x_np = x.astype(dtype, casting=casting, copy=copy)
            return self._namespace.asarray(x_np)

        f = self._namespace.astype
        return f(x, dtype, copy=copy)

    def asarray(self, obj, *, dtype=None, device=None, copy=None, order=None):
        # support order in NumPy
        if self._namespace.__name__ == "numpy.aray_api":
            if copy:
                x_np = numpy.array(obj, dtype=dtype, order=order, copy=True)
            else:
                x_np = numpy.asarray(obj, dtype=dtype, order=order)
            return self._namespace(x_np)

        f = self._namespace.asarray
        return f(obj, dtype=dtype, device=device, copy=copy)

    def may_share_memory(self, a, b):
        # support may_share_memory in NumPy
        if self._namespace.__name__ == "numpy.aray_api":
            return numpy.may_share_memory(a, b)

        # The safe choice is to return True for all other array_api Arrays
        return True


class _NumPyApiWrapper:
    def __getattr__(self, name):
        return getattr(numpy, name)

    def astype(self, x, dtype, *, copy=True, casting="unsafe"):
        # astype is not defined in the top level NumPy namespace
        return x.astype(dtype, copy=copy, casting=casting)

    def asarray(self, obj, *, dtype=None, device=None, copy=None, order=None):
        # copy is in the ArrayAPI spec but not in NumPy's asarray
        if copy:
            return numpy.array(obj, dtype=dtype, order=order, copy=True)
        else:
            return numpy.asarray(obj, dtype=dtype, order=order)


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
    xp, is_array_api = get_namespace(a)

    # Use SciPy if a is an ndarray
    if not is_array_api:
        return scipy.special.logsumexp(
            a, axis=axis, b=b, keepdims=keepdims, return_sign=return_sign
        )

    if b is not None:
        a, b = xp.broadcast_arrays(a, b)
        if xp.any(b == 0):
            a = a + 0.0  # promote to at least float
            a[b == 0] = -xp.inf

    a_max = xp.max(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~xp.isfinite(a_max)] = 0
    elif not xp.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = xp.asarray(b)
        tmp = b * xp.exp(a - a_max)
    else:
        tmp = xp.exp(a - a_max)

    # suppress warnings about log of zero
    s = xp.sum(tmp, axis=axis, keepdims=keepdims)
    if return_sign:
        sgn = xp.sign(s)
        s *= sgn  # /= makes more sense but we need zero -> zero
    out = xp.log(s)

    if not keepdims:
        a_max = xp.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out
