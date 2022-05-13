from functools import wraps
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import issparse


class SKSparseMixin:
    def __init__(self, *args, columns=None, index=None, **kwargs):
        super().__init__(*args, **kwargs)
        if columns is not None:
            self.columns = columns
        if index is not None:
            self.index = index


class SKCSRMatrix(SKSparseMixin, csr_matrix):
    pass


class SKCSCMatrix(SKSparseMixin, csc_matrix):
    pass


def _get_output_type_config(self, key):
    if hasattr(self, "_output_types"):
        output_types = self._output_types
    else:
        output_types = {"transform": {"dense": "default", "sparse": "default"}}
    return output_types[key]


def _wrap_output(self, X, output, dtype=None):
    output_type = _get_output_type_config(self, "transform")

    if output_type["dense"] == "default" or output_type["sparse"] == "default":
        return output

    if hasattr(output, "iloc"):
        return output

    X_is_sparse = issparse(X)
    if X_is_sparse and output_type["sparse"] == "default":
        return output
    if not X_is_sparse and output_type["dense"] == "default":
        return output

    names_out = self.get_feature_names_out()
    index = getattr(X, "index", None)

    if isinstance(output, csr_matrix):
        return SKCSRMatrix(output, columns=names_out, index=index)
    elif isinstance(output, csc_matrix):
        return SKCSCMatrix(output, columns=names_out, index=index)

    import pandas as pd

    return pd.DataFrame(output, columns=names_out, index=index, dtype=dtype)


def wrap_output_f(f):
    @wraps(f)
    def wrapped(self, X, *args, **kwargs):
        output = f(self, X, *args, **kwargs)
        return _wrap_output(self, X, output)

    return wrapped


def _set_outout_helper(self, transform=None):
    if not hasattr(self, "_output_types"):
        self._output_types = {}

    if isinstance(transform, str):
        if "/" in transform:
            dense, sparse = transform.split("_or_")
        else:
            dense, sparse = transform, transform

    self._output_types["transform"] = {"dense": dense, "sparse": sparse}


class OutputTypeMixin:
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "transform"):
            cls.transform = wrap_output_f(cls.transform)
        if hasattr(cls, "fit_transform"):
            cls.fit_transform = wrap_output_f(cls.fit_transform)

    def set_output(self, transform=None):
        if transform is None:
            return self
        _set_outout_helper(self, transform=transform)
        return self


def _safe_set_output(est, warn=True, **kwargs):
    if not hasattr(est, "set_output"):
        if warn:
            warnings.warn(
                f"Unable to configure output for {est} because `set_output` "
                "is not avaliable."
            )
        return

    return est.set_output(**kwargs)
