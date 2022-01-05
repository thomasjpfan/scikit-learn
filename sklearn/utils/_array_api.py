"""Tools to support array_api."""
import numpy as np
from scipy.special import logsumexp as sp_logsumexp
from .._config import get_config


def get_namespace(*xs):
    # `xs` contains one or more arrays, or possibly Python scalars (accepting
    # those is a matter of taste, but doesn't seem unreasonable).
    # Returns a tuple: (array_namespace, is_array_api)

    if not get_config()["array_api_dispatch"]:
        return np, False

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
        return np, False

    return xp, True


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
