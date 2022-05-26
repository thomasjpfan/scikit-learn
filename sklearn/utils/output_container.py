from functools import wraps
import warnings

from scipy.sparse import bsr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import issparse

from sklearn.utils import check_pandas_support
from sklearn._config import get_config


class SKSparseMixin:
    """Scikit-learn Sparse matrix mixin.

    Parameters
    ----------
    *args : tuple
        Arguments to pass to SciPy's sparse matrix constructor.

    columns : array, default=None
        If not `None`, set `columns` as an attribute.

    index : array, default=None
        If not `None`, set `index` as an attribute.

    **kwargs : dict
        Keyword arguments to pass to SciPy's sparse matrix constructor.
    """

    def __init__(self, *args, columns=None, index=None, **kwargs):
        super().__init__(*args, **kwargs)
        if columns is not None:
            self.columns = columns
        if index is not None:
            self.index = index


class NamedBSRMatrix(SKSparseMixin, bsr_matrix):
    pass


class NamedCOOMatrix(SKSparseMixin, coo_matrix):
    pass


class NamedCSCMatrix(SKSparseMixin, csc_matrix):
    pass


class NamedCSRMatrix(SKSparseMixin, csr_matrix):
    pass


class NamedDIAMatrix(SKSparseMixin, dia_matrix):
    pass


class NamedDOKMatrix(SKSparseMixin, dok_matrix):
    pass


class NamedLILMatrix(SKSparseMixin, lil_matrix):
    pass


_namedsparse_classes = set(
    [
        NamedBSRMatrix,
        NamedCOOMatrix,
        NamedCSCMatrix,
        NamedCSRMatrix,
        NamedDIAMatrix,
        NamedDOKMatrix,
        NamedLILMatrix,
    ]
)

for kls in _namedsparse_classes:
    kls.__doc__ = SKSparseMixin.__doc__

_sparse_to_namedsparse = {mat.format: mat for mat in _namedsparse_classes}


def make_named_container(
    output,
    *,
    get_columns=None,
    dense_container="default",
    sparse_container="default",
    **kwargs,
):
    """Create a named container.

    Parameters
    ----------
    output : ndarray, sparse matrix or pandas DataFrame
        Container to name.

    get_columns : callable, default=None
        Callable to get column names.

    dense_container : {"default", "pandas"}, default="default"
        Container used for dense data.

    sparse_container : {"default", "namedsparse"}, default="default"
        Container used for sparse data.

    **kwargs : dict
        Keyword arguments passed to container constructor.

    Returns
    -------
    output : DataFrame or Named Sparse Matrix
        Container with column names.
    """
    if dense_container not in {"default", "pandas"}:
        raise ValueError(
            f"dense_container must be 'default' or 'pandas' got {dense_container}"
        )
    if sparse_container not in {"default", "namedsparse"}:
        raise ValueError(
            "sparse_container must be 'default' or 'namedsparse' got"
            f" {sparse_container}"
        )

    if issparse(output):
        if sparse_container == "default":
            return output
        # assume sparse_container == "namedsparse"

        # noop if output is already a named sparse object
        if hasattr(output, "columns"):
            # XXX: Should we override the feature names?
            # if get_columns is not None:
            #     output.columns = get_columns()
            return output

        SparseContainer = _sparse_to_namedsparse[output.getformat()]
        if get_columns is not None:
            kwargs["columns"] = get_columns()
        return SparseContainer(output, **kwargs)

    if dense_container == "default":
        return output

    # noop if output is already a DataFrame
    if hasattr(output, "columns"):
        # XXX: Should we override the feature names?
        # if get_columns is not None:
        #     output.columns = get_columns()
        return output

    # assume dense_container == "pandas"
    pd = check_pandas_support("make_named_container")
    if get_columns is not None:
        kwargs["columns"] = get_columns()
    return pd.DataFrame(output, **kwargs)


def get_output_container_config(method, estimator=None):
    """Get output configure based on estimator and global configuration.

    Parameters
    ----------
    method : str
        Method to get container output for.

    estimator : estimator instance, default=None
        If not `None`, check the estimator for output container.

    Returns
    -------
    config : dict
        Dictionary with keys, "dense" and "sparse", that specifies the
        container for `method`.
    """
    est_output_containers = getattr(estimator, "_output_containers", {})
    if method in est_output_containers:
        container_str = est_output_containers[method]
    else:
        container_str = get_config()[f"output_{method}"]

    if "_or_" in container_str:
        dense_container, sparse_container = container_str.split("_or_")
    else:
        dense_container = sparse_container = container_str

    return {"dense": dense_container, "sparse": sparse_container}


def _wrap_output(estimator, X, output, method, **kwargs):
    output_container = get_output_container_config(method, estimator=estimator)

    # cross decomposition outputs a tuple, Here we only wrap the first output
    if isinstance(output, tuple):
        output = output[0]
    return make_named_container(
        output=output,
        get_columns=getattr(estimator, "get_feature_names_out", None),
        dense_container=output_container["dense"],
        sparse_container=output_container["sparse"],
        index=getattr(X, "index", None),
        **kwargs,
    )


def _wrap_output_f(f, method):
    @wraps(f)
    def wrapped(self, X, *args, **kwargs):
        output = f(self, X, *args, **kwargs)
        return _wrap_output(self, X, output, method)

    return wrapped


def set_output_helper(estimator, transform=None):
    """Set output for estimator.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance.

    transform : {"default", "pandas_or_namedsparse"}, default=None
        Configure output of `transform` and `fit_transform`.
    """
    if transform is None:
        return

    if not hasattr(estimator, "_output_containers"):
        estimator._output_containers = {}

    estimator._output_containers["transform"] = transform


class OutputTypeMixin:
    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, "transform"):
            cls.transform = _wrap_output_f(cls.transform, "transform")
        if hasattr(cls, "fit_transform"):
            cls.fit_transform = _wrap_output_f(cls.fit_transform, "transform")

    def set_output(self, transform=None):
        """Set output container.

        Parameters
        ----------
        transform : {"default", "pandas_or_namedsparse"}, default=None
            Configure output of `transform` and `fit_transform`.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        set_output_helper(self, transform=transform)
        return self


def safe_set_output(estimator, warn=True, transform=None):
    """Safely call estimator.set_output and warn when needed.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance.

    warn : bool, default=True
        If `True`, warn if `estimator.set_output` is not defined.

    transform : {"default", "pandas_or_namedsparse"}, default=None
        Configure output of `transform` and `fit_transform`.

    Returns
    -------
    estimator : estimator instance
        Estimator instance.
    """
    if not hasattr(estimator, "set_output"):
        has_transform = hasattr(estimator, "transform") or hasattr(
            estimator, "fit_transform"
        )
        if warn and transform is not None and has_transform:
            warnings.warn(
                f"Unable to configure output for {estimator} because `set_output` "
                "is not available"
            )
        return
    return estimator.set_output(transform=transform)
