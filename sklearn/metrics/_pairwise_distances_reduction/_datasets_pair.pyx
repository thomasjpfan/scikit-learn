import numpy as np
cimport numpy as cnp

from cython cimport final
from scipy.sparse import issparse, csr_matrix

from ...utils._typedefs cimport DTYPE_t, ITYPE_t
from ...metrics._dist_metrics cimport DistanceMetric

from ...utils._typedefs import DTYPE, SPARSE_INDEX_TYPE


cnp.import_array()

cdef class DatasetsPair:
    """Abstract class which wraps a pair of datasets (X, Y).

    This class allows computing distances between a single pair of rows of
    of X and Y at a time given the pair of their indices (i, j). This class is
    specialized for each metric thanks to the :func:`get_for` factory classmethod.

    The handling of parallelization over chunks to compute the distances
    and aggregation for several rows at a time is done in dedicated
    subclasses of PairwiseDistancesReduction that in-turn rely on
    subclasses of DatasetsPair for each pair of rows in the data. The goal
    is to make it possible to decouple the generic parallelization and
    aggregation logic from metric-specific computation as much as
    possible.

    X and Y can be stored as C-contiguous np.ndarrays or CSR matrices
    in subclasses.

    This class avoids the overhead of dispatching distance computations
    to :class:`sklearn.metrics.DistanceMetric` based on the physical
    representation of the vectors (sparse vs. dense). It makes use of
    cython.final to remove the overhead of dispatching method calls.

    Parameters
    ----------
    distance_metric: DistanceMetric
        The distance metric responsible for computing distances
        between two vectors of (X, Y).
    """

    @classmethod
    def get_for(
        cls,
        X,
        Y,
        str metric="euclidean",
        dict metric_kwargs=None,
    ) -> DatasetsPair:
        """Return the DatasetsPair implementation for the given arguments.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.
            If provided as a ndarray, it must be C-contiguous.
            If provided as a sparse matrix, it must be in CSR format.

        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)
            Input data.
            If provided as a ndarray, it must be C-contiguous.
            If provided as a sparse matrix, it must be in CSR format.

        metric : str, default='euclidean'
            The distance metric to compute between rows of X and Y.
            The default metric is a fast implementation of the Euclidean
            metric. For a list of available metrics, see the documentation
            of :class:`~sklearn.metrics.DistanceMetric`.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        Returns
        -------
        datasets_pair: DatasetsPair
            The suited DatasetsPair implementation.
        """
        cdef:
            DistanceMetric distance_metric = DistanceMetric.get_metric(
                metric,
                **(metric_kwargs or {})
            )

        if not(X.dtype == Y.dtype == np.float64):
            raise ValueError(
                f"Only 64bit float datasets are supported at this time, "
                f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
            )

        # Metric-specific checks that do not replace nor duplicate `check_array`.
        distance_metric._validate_data(X)
        distance_metric._validate_data(Y)

        X_is_sparse = issparse(X)
        Y_is_sparse = issparse(Y)

        if not X_is_sparse and not Y_is_sparse:
            return DenseDenseDatasetsPair(X, Y, distance_metric)

        if X_is_sparse and Y_is_sparse:
            return SparseSparseDatasetsPair(X, Y, distance_metric)

        if X_is_sparse and not Y_is_sparse:
            return SparseDenseDatasetsPair(X, Y, distance_metric)

        return DenseSparseDatasetsPair(X, Y, distance_metric)

    @classmethod
    def unpack_csr_matrix(cls, X: csr_matrix):
        """Ensure that the CSR matrix is indexed with SPARSE_INDEX_TYPE."""
        # TODO: leave X.data unchanged once float32 is supported.
        X_data = np.asarray(X.data, dtype=DTYPE)
        X_indices = np.asarray(X.indices, dtype=SPARSE_INDEX_TYPE)
        X_indptr = np.asarray(X.indptr, dtype=SPARSE_INDEX_TYPE)
        return X_data, X_indices, X_indptr

    def __init__(self, DistanceMetric distance_metric, ITYPE_t n_features):
        self.distance_metric = distance_metric
        self.n_features = n_features

    cdef ITYPE_t n_samples_X(self) nogil:
        """Number of samples in X."""
        # This is a abstract method.
        # This _must_ always be overwritten in subclasses.
        # TODO: add "with gil: raise" here when supporting Cython 3.0
        return -999

    cdef ITYPE_t n_samples_Y(self) nogil:
        """Number of samples in Y."""
        # This is a abstract method.
        # This _must_ always be overwritten in subclasses.
        # TODO: add "with gil: raise" here when supporting Cython 3.0
        return -999

    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.dist(i, j)

    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil:
        # This is a abstract method.
        # This _must_ always be overwritten in subclasses.
        # TODO: add "with gil: raise" here when supporting Cython 3.0
        return -1

@final
cdef class DenseDenseDatasetsPair(DatasetsPair):
    """Compute distances between row vectors of two arrays.

    Parameters
    ----------
    X: ndarray of shape (n_samples_X, n_features)
        Rows represent vectors. Must be C-contiguous.

    Y: ndarray of shape (n_samples_Y, n_features)
        Rows represent vectors. Must be C-contiguous.

    distance_metric: DistanceMetric
        The distance metric responsible for computing distances
        between two row vectors of (X, Y).
    """

    def __init__(
        self,
        const DTYPE_t[:, ::1] X,
        const DTYPE_t[:, ::1] Y,
        DistanceMetric distance_metric,
    ):
        super().__init__(distance_metric, n_features=X.shape[1])
        # Arrays have already been checked
        self.X = X
        self.Y = Y

    @final
    cdef ITYPE_t n_samples_X(self) nogil:
        return self.X.shape[0]

    @final
    cdef ITYPE_t n_samples_Y(self) nogil:
        return self.Y.shape[0]

    @final
    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.distance_metric.rdist(&self.X[i, 0], &self.Y[j, 0], self.n_features)

    @final
    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.distance_metric.dist(&self.X[i, 0], &self.Y[j, 0], self.n_features)


@final
cdef class SparseSparseDatasetsPair(DatasetsPair):
    """Compute distances between vectors of two CSR matrices.

    Parameters
    ----------
    X: sparse matrix of shape (n_samples_X, n_features)
        Rows represent vectors. Must be in CSR format.

    Y: sparse matrix of shape (n_samples_Y, n_features)
        Rows represent vectors. Must be in CSR format.

    distance_metric: DistanceMetric
        The distance metric responsible for computing distances
        between two vectors of (X, Y).
    """

    def __init__(self, X, Y, DistanceMetric distance_metric):
        super().__init__(distance_metric, n_features=X.shape[1])

        self.X_data, self.X_indices, self.X_indptr = self.unpack_csr_matrix(X)
        self.Y_data, self.Y_indices, self.Y_indptr = self.unpack_csr_matrix(Y)

    @final
    cdef ITYPE_t n_samples_X(self) nogil:
        return self.X_indptr.shape[0] - 1

    @final
    cdef ITYPE_t n_samples_Y(self) nogil:
        return self.Y_indptr.shape[0] -1

    @final
    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.distance_metric.rdist_csr(
            x1_data=self.X_data,
            x1_indices=self.X_indices,
            x2_data=self.Y_data,
            x2_indices=self.Y_indices,
            x1_start=self.X_indptr[i],
            x1_end=self.X_indptr[i + 1],
            x2_start=self.Y_indptr[j],
            x2_end=self.Y_indptr[j + 1],
            size=self.n_features,
        )

    @final
    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.distance_metric.dist_csr(
            x1_data=self.X_data,
            x1_indices=self.X_indices,
            x2_data=self.Y_data,
            x2_indices=self.Y_indices,
            x1_start=self.X_indptr[i],
            x1_end=self.X_indptr[i + 1],
            x2_start=self.Y_indptr[j],
            x2_end=self.Y_indptr[j + 1],
            size=self.n_features,
        )


@final
cdef class SparseDenseDatasetsPair(DatasetsPair):
    """Compute distances between vectors of a CSR matrix and a dense array.

    Parameters
    ----------
    X: sparse matrix of shape (n_samples_X, n_features)
        Rows represent vectors. Must be in CSR format.

    Y: ndarray of shape (n_samples_Y, n_features)
        Rows represent vectors. Must be C-contiguous.

    distance_metric: DistanceMetric
        The distance metric responsible for computing distances
        between two vectors of (X, Y).
    """

    def __init__(self, X, Y, DistanceMetric distance_metric):
        super().__init__(distance_metric, n_features=X.shape[1])

        self.X_data, self.X_indices, self.X_indptr = self.unpack_csr_matrix(X)

        # Y array already has been checked here
        self.n_Y = Y.shape[0]
        self.Y_data = np.ravel(Y)

        # Since Y vectors are dense, we can use a single array
        # of indices of self.n_features elements instead of
        # a self.n_Y × self.n_features matrix.
        # The implementations of DistanceMetric.{dist_csr,rdist_csr}
        # support this representation.
        self.Y_indices = np.arange(self.n_features, dtype=SPARSE_INDEX_TYPE)

    @final
    cdef ITYPE_t n_samples_X(self) nogil:
        return self.X_indptr.shape[0] - 1

    @final
    cdef ITYPE_t n_samples_Y(self) nogil:
        return self.n_Y

    @final
    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.distance_metric.rdist_csr(
            x1_data=self.X_data,
            x1_indices=self.X_indices,
            x2_data=self.Y_data,
            x2_indices=self.Y_indices,
            x1_start=self.X_indptr[i],
            x1_end=self.X_indptr[i + 1],
            x2_start=j * self.n_features,
            x2_end=(j + 1) * self.n_features,
            size=self.n_features,
        )

    @final
    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil:
        return self.distance_metric.dist_csr(
            x1_data=self.X_data,
            x1_indices=self.X_indices,
            x2_data=self.Y_data,
            x2_indices=self.Y_indices,
            x1_start=self.X_indptr[i],
            x1_end=self.X_indptr[i + 1],
            x2_start=j * self.n_features,
            x2_end=(j + 1) * self.n_features,
            size=self.n_features,
        )


@final
cdef class DenseSparseDatasetsPair(DatasetsPair):
    """Compute distances between vectors of a dense array and a CSR matrix.

    Parameters
    ----------
    X: ndarray of shape (n_samples_X, n_features)
        Rows represent vectors. Must be C-contiguous.

    Y: sparse matrix of shape (n_samples_Y, n_features)
        Rows represent vectors. Must be in CSR format.

    distance_metric: DistanceMetric
        The distance metric responsible for computing distances
        between two vectors of (X, Y).
    """

    def __init__(self, X, Y, DistanceMetric distance_metric):
        super().__init__(distance_metric, n_features=X.shape[1])
        # Swapping arguments on the constructor
        self.datasets_pair = SparseDenseDatasetsPair(Y, X, distance_metric)

    @final
    cdef ITYPE_t n_samples_X(self) nogil:
        # Swapping interface
        return self.datasets_pair.n_samples_Y()

    @final
    cdef ITYPE_t n_samples_Y(self) nogil:
        # Swapping interface
        return self.datasets_pair.n_samples_X()

    @final
    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil:
        # Swapping arguments on the same interface
        return self.datasets_pair.surrogate_dist(j, i)

    @final
    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil:
        # Swapping arguments on the same interface
        return self.datasets_pair.dist(j, i)
