# By Jake Vanderplas (2013) <jakevdp@cs.washington.edu>
# written for the scikit-learn project
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
cimport numpy as cnp

cnp.import_array()  # required in order to use C-API

from libc.stdint cimport uintptr_t

from cython cimport floating
from scipy.sparse import csr_matrix, issparse
from sklearn.utils._typedefs cimport float64_t, float32_t, int32_t, intp_t
from sklearn.utils import check_array
from sklearn.utils.fixes import parse_version, sp_base_version

# The vector-wise distance math lives once, in C++, in src/metric_kernels.hpp.
# The single ``DistanceMetric`` class below is a thin holder that delegates its
# cdef methods to a ``MetricBase[T]`` functor built by the make_* factories, for
# the dtype it was configured with (float32 vs float64).
cdef extern from "metric_kernels.hpp" namespace "sklearn::metrics" nogil:
    cdef cppclass MetricBase[T]:
        T dist(const T* x1, const T* x2, intp_t size) noexcept
        T rdist(const T* x1, const T* x2, intp_t size) noexcept
        T dist_csr(const T* x1_data, const int32_t* x1_indices,
                   const T* x2_data, const int32_t* x2_indices,
                   int32_t x1_start, int32_t x1_end,
                   int32_t x2_start, int32_t x2_end, intp_t size) noexcept
        T rdist_csr(const T* x1_data, const int32_t* x1_indices,
                    const T* x2_data, const int32_t* x2_indices,
                    int32_t x1_start, int32_t x1_end,
                    int32_t x2_start, int32_t x2_end, intp_t size) noexcept
        T rdist_to_dist(T rdist) noexcept
        T dist_to_rdist(T dist) noexcept

# Factories return owning raw pointers; the holder deletes its functor on dealloc.
# These may allocate, so they keep the GIL-reacquiring `except +` translation.
cdef extern from "metric_kernels.hpp" namespace "sklearn::metrics":
    MetricBase[T]* make_euclidean[T]() except +
    MetricBase[T]* make_seuclidean[T](const double* V, intp_t n) except +
    MetricBase[T]* make_manhattan[T]() except +
    MetricBase[T]* make_chebyshev[T]() except +
    MetricBase[T]* make_minkowski[T](double p, const double* w, intp_t w_len) except +
    MetricBase[T]* make_mahalanobis[T](const double* VI, intp_t n) except +
    MetricBase[T]* make_hamming[T]() except +
    MetricBase[T]* make_canberra[T]() except +
    MetricBase[T]* make_braycurtis[T]() except +
    MetricBase[T]* make_jaccard[T]() except +
    MetricBase[T]* make_matching[T]() except +
    MetricBase[T]* make_dice[T]() except +
    MetricBase[T]* make_kulsinski[T]() except +
    MetricBase[T]* make_rogerstanimoto[T]() except +
    MetricBase[T]* make_russellrao[T]() except +
    MetricBase[T]* make_sokalmichener[T]() except +
    MetricBase[T]* make_sokalsneath[T]() except +
    MetricBase[T]* make_haversine[T]() except +


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


BOOL_METRICS = [
    "hamming",
    "jaccard",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalsneath",
]
DEPRECATED_METRICS = []
if sp_base_version < parse_version("1.17"):
    # Deprecated in SciPy 1.15 and removed in SciPy 1.17
    BOOL_METRICS += ["sokalmichener"]
if sp_base_version >= parse_version("1.15"):
    DEPRECATED_METRICS.append("sokalmichener")
if sp_base_version < parse_version("1.11"):
    # Deprecated in SciPy 1.9 and removed in SciPy 1.11
    BOOL_METRICS += ["kulsinski"]
if sp_base_version >= parse_version("1.9"):
    DEPRECATED_METRICS.append("kulsinski")
if sp_base_version < parse_version("1.9"):
    # Deprecated in SciPy 1.0 and removed in SciPy 1.9
    BOOL_METRICS += ["matching"]
if sp_base_version >= parse_version("1.0"):
    DEPRECATED_METRICS.append("matching")


def get_valid_metric_ids(L):
    """Given an iterable of metric class names or class identifiers,
    return a list of metric IDs which map to those classes.

    Example:
    >>> L = get_valid_metric_ids([EuclideanDistance, 'ManhattanDistance'])
    >>> sorted(L)
    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    """
    return [key for (key, val) in METRIC_MAPPING.items()
            if (val.__name__ in L) or (val in L)]


cdef inline object _buffer_to_ndarray64(const float64_t* x, intp_t n):
    # Wrap a memory buffer with an ndarray. Warning: this is not robust.
    # In particular, if x is deallocated before the returned array goes
    # out of scope, this could cause memory errors.  Since there is not
    # a possibility of this for our use-case, this should be safe.

    # Note: this Segfaults unless np.import_array() is called above
    return cnp.PyArray_SimpleNewFromData(1, <cnp.intp_t*>&n, cnp.NPY_FLOAT64, <void*>x)


cdef inline object _buffer_to_ndarray32(const float32_t* x, intp_t n):
    return cnp.PyArray_SimpleNewFromData(1, <cnp.intp_t*>&n, cnp.NPY_FLOAT32, <void*>x)


cdef float64_t INF = np.inf


######################################################################
# Distance Metric Classes
cdef class DistanceMetric:
    """Uniform interface for fast distance metric functions.

    The `DistanceMetric` class provides a convenient way to compute pairwise distances
    between samples. It supports various distance metrics, such as Euclidean distance,
    Manhattan distance, and more.

    The `pairwise` method can be used to compute pairwise distances between samples in
    the input arrays. It returns a distance matrix representing the distances between
    all pairs of samples.

    The :meth:`get_metric` method allows you to retrieve a specific metric using its
    string identifier.

    Examples
    --------
    >>> from sklearn.metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> Y = [[7, 8], [9, 10]]
    >>> dist.pairwise(X,Y)
    array([[7.81..., 10.63...]
           [5.65...,  8.48...]
           [1.41...,  4.24...]])

    .. rubric:: Available Metrics

    The following lists the string metric identifiers and the associated
    distance metric classes:

    **Metrics intended for real-valued vector spaces:**

    ==============  ====================  ========  ===============================
    identifier      class name            args      distance function
    --------------  --------------------  --------  -------------------------------
    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``
    "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``
    "minkowski"     MinkowskiDistance     p, w      ``sum(w * |x - y|^p)^(1/p)``
    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``
    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)' V^-1 (x - y))``
    ==============  ====================  ========  ===============================

    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine
    distance metric requires data in the form of [latitude, longitude] and both
    inputs and outputs are in units of radians.

    ============  ==================  ===============================================================
    identifier    class name          distance function
    ------------  ------------------  ---------------------------------------------------------------
    "haversine"   HaversineDistance   ``2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))``
    ============  ==================  ===============================================================


    **Metrics intended for integer-valued vector spaces:**  Though intended
    for integer-valued vectors, these are also valid metrics in the case of
    real-valued vectors.

    =============  ====================  ========================================
    identifier     class name            distance function
    -------------  --------------------  ----------------------------------------
    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``
    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``
    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``
    =============  ====================  ========================================

    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry
    is evaluated to "True".  In the listings below, the following
    abbreviations are used:

    - N: number of dimensions
    - NTT: number of dims in which both values are True
    - NTF: number of dims in which the first value is True, second is False
    - NFT: number of dims in which the first value is False, second is True
    - NFF: number of dims in which both values are False
    - NNEQ: number of non-equal dimensions, NNEQ = NTF + NFT
    - NNZ: number of nonzero dimensions, NNZ = NTF + NFT + NTT

    =================  =======================  ===============================
    identifier         class name               distance function
    -----------------  -----------------------  -------------------------------
    "jaccard"          JaccardDistance          NNEQ / NNZ
    "matching"         MatchingDistance         NNEQ / N
    "dice"             DiceDistance             NNEQ / (NTT + NNZ)
    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)
    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)
    "russellrao"       RussellRaoDistance       (N - NTT) / N
    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)
    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)
    =================  =======================  ===============================

    **User-defined distance:**

    ===========    ===============    =======
    identifier     class name         args
    -----------    ---------------    -------
    "pyfunc"       PyFuncDistance     func
    ===========    ===============    =======

    Here ``func`` is a function which takes two one-dimensional numpy
    arrays, and returns a distance.  Note that in order to be used within
    the BallTree, the distance must be a true metric:
    i.e. it must satisfy the following properties

    1) Non-negativity: d(x, y) >= 0
    2) Identity: d(x, y) = 0 if and only if x == y
    3) Symmetry: d(x, y) = d(y, x)
    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

    Because of the Python object overhead involved in calling the python
    function, this will be fairly slow, but it will have the same
    scaling as other distances.
    """
    def __cinit__(self):
        self.p = 2
        self.vec = np.zeros(1, dtype=np.float64, order='C')
        self.mat = np.zeros((1, 1), dtype=np.float64, order='C')
        self.size = 1
        self.functor = NULL
        self._use_f32 = False

    def __dealloc__(self):
        self._free_functor()

    def __init__(self):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")
        # Parameterless metrics rely on this default to build their functor;
        # metrics with parameters override __init__ and call _build_functor last.
        self._build_functor()

    @classmethod
    def get_metric(cls, metric, dtype=np.float64, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : str or class name
            The string identifier or class name of the desired distance metric.
            See the documentation of the `DistanceMetric` class for a list of
            available metrics.

        dtype : {np.float32, np.float64}, default=np.float64
            The data type of the input on which the metric will be applied.
            This affects the precision of the computed distances.
            By default, it is set to `np.float64`.

        **kwargs
            Additional keyword arguments that will be passed to the requested metric.
            These arguments can be used to customize the behavior of the specific
            metric.

        Returns
        -------
        metric_obj : instance of the requested metric
            An instance of the requested distance metric class.
        """
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                f"Unexpected dtype {dtype} provided. Please select a dtype from"
                " {np.float32, np.float64}"
            )

        if isinstance(metric, DistanceMetric):
            return metric

        if callable(metric):
            obj = PyFuncDistance(metric, **kwargs)
        else:
            # Map the metric string ID to the metric class
            if isinstance(metric, type) and issubclass(metric, DistanceMetric):
                pass
            else:
                try:
                    metric = METRIC_MAPPING[metric]
                except Exception:
                    raise ValueError("Unrecognized metric '%s'" % metric)

            # In Minkowski special cases, return more efficient methods
            if metric is MinkowskiDistance:
                p = kwargs.pop('p', 2)
                w = kwargs.pop('w', None)
                if p == 1 and w is None:
                    obj = ManhattanDistance(**kwargs)
                elif p == 2 and w is None:
                    obj = EuclideanDistance(**kwargs)
                elif np.isinf(p) and w is None:
                    obj = ChebyshevDistance(**kwargs)
                else:
                    obj = MinkowskiDistance(p, w, **kwargs)
            else:
                obj = metric(**kwargs)

        if dtype == np.float32:
            obj._set_float32()
        return obj

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        if isinstance(self, PyFuncDistance):
            return (float(self.p), np.asarray(self.vec), np.asarray(self.mat),
                    self.func, self.kwargs, bool(self._use_f32))
        return (float(self.p), np.asarray(self.vec), np.asarray(self.mat),
                bool(self._use_f32))

    def __setstate__(self, state):
        """
        set state for pickling
        """
        cdef bint use_f32 = False
        self.p = state[0]
        self.vec = state[1]
        self.mat = state[2]
        # Tuple length disambiguates the (possibly older) state layout:
        #   3 -> (p, vec, mat)                     [legacy, float64]
        #   4 -> (p, vec, mat, use_f32)
        #   5 -> (p, vec, mat, func, kwargs)       [legacy PyFunc, float64]
        #   6 -> (p, vec, mat, func, kwargs, use_f32)
        if len(state) == 4:
            use_f32 = state[3]
        elif len(state) == 5:
            self.func = state[3]
            self.kwargs = state[4]
        elif len(state) == 6:
            self.func = state[3]
            self.kwargs = state[4]
            use_f32 = state[5]
        self.size = self.vec.shape[0]
        # Rebuild the functor from the just-restored parameters and dtype.
        self._free_functor()
        self._use_f32 = use_f32
        self._build_functor()

    def _set_float32(self):
        """Switch this metric to operate on float32 inputs (rebuilds the functor)."""
        if self._use_f32:
            return
        self._free_functor()
        self._use_f32 = True
        self._build_functor()

    def _functor_address(self):
        """Address of the C++ MetricBase functor, as a Python int.

        Used by the C++ pairwise-distances reductions to borrow the functor for
        the generic-metric path. The owning DistanceMetric instance must be kept
        alive for the whole duration of the borrow.
        """
        return <uintptr_t>self.functor

    def _validate_data(self, X):
        """Validate the input data.

        This should be overridden in a base class if a specific input format
        is required.
        """
        return

    cdef void _build_functor(self) except *:
        """Create ``self.functor`` from the instance parameters (overridden)."""
        raise NotImplementedError(
            "DistanceMetric subclasses must implement _build_functor"
        )

    cdef void _free_functor(self) noexcept:
        cdef MetricBase[float32_t]* f32
        cdef MetricBase[float64_t]* f64
        if self.functor is NULL:
            return
        if self._use_f32:
            f32 = <MetricBase[float32_t]*>self.functor
            del f32
        else:
            f64 = <MetricBase[float64_t]*>self.functor
            del f64
        self.functor = NULL

    cdef float64_t dist(
        self,
        const floating* x1,
        const floating* x2,
        intp_t size,
    ) except -1 nogil:
        """Compute the distance between vectors x1 and x2."""
        if self.functor is NULL:
            with gil:
                return self._pyfunc_dist(x1, x2, size)
        return (<MetricBase[floating]*>self.functor).dist(x1, x2, size)

    cdef float64_t rdist(
        self,
        const floating* x1,
        const floating* x2,
        intp_t size,
    ) except -1 nogil:
        """Compute the rank-preserving surrogate distance between x1 and x2.

        The rank-preserving surrogate distance is any measure that yields the same
        rank as the distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.
        """
        if self.functor is NULL:
            with gil:
                return self._pyfunc_dist(x1, x2, size)
        return (<MetricBase[floating]*>self.functor).rdist(x1, x2, size)

    cdef float64_t _pyfunc_dist(
        self,
        const floating* x1,
        const floating* x2,
        intp_t size,
    ) except -1 with gil:
        cdef object x1arr
        cdef object x2arr
        if floating is float:
            x1arr = _buffer_to_ndarray32(x1, size)
            x2arr = _buffer_to_ndarray32(x2, size)
        else:
            x1arr = _buffer_to_ndarray64(x1, size)
            x2arr = _buffer_to_ndarray64(x2, size)
        d = self.func(x1arr, x2arr, **self.kwargs)
        try:
            # Cython generates code here that results in a TypeError
            # if d is the wrong type.
            return d
        except TypeError:
            raise TypeError("Custom distance function must accept two "
                            "vectors and return a float.")

    cdef float64_t dist_csr(
        self,
        const floating* x1_data,
        const int32_t* x1_indices,
        const floating* x2_data,
        const int32_t* x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        """Compute the distance between vectors x1 and x2, both in CSR format.

        Notes
        -----
        0. The implementation of this method in subclasses must be robust to the
        presence of explicit zeros in the CSR representation.

        1. The `data` arrays are passed using pointers to be able to support an
        alternative representation of the CSR data structure for supporting
        fused sparse-dense datasets pairs with minimum overhead.

        See the explanations in `SparseDenseDatasetsPair.__init__`.

        2. The `x{1,2}_{data,indices}` arrays are passed as well as their indices
        pointers (`x{1,2}_{start,end}`) to avoid slicing the data and indices
        arrays of the sparse matrices, which would take the GIL.
        See: https://github.com/scikit-learn/scikit-learn/issues/17299

        3. For reference about the CSR format, see section 3.4 of
        Saad, Y. (2003), Iterative Methods for Sparse Linear Systems, SIAM.
        https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
        """
        if self.functor is NULL:
            # PyFunc and the abstract base do not support the CSR interface.
            return -999
        return (<MetricBase[floating]*>self.functor).dist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, size,
        )

    cdef float64_t rdist_csr(
        self,
        const floating* x1_data,
        const int32_t* x1_indices,
        const floating* x2_data,
        const int32_t* x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        """Rank-preserving surrogate distance between rows of CSR matrices.

        More information about the motives for this method signature is given
        in the docstring of dist_csr.
        """
        if self.functor is NULL:
            return -999
        return (<MetricBase[floating]*>self.functor).rdist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, size,
        )

    cdef int pdist(
        self,
        const floating[:, ::1] X,
        floating[:, ::1] D,
    ) except -1:
        """Compute the pairwise distances between points in X"""
        cdef intp_t i1, i2
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return 0

    cdef int cdist(
        self,
        const floating[:, ::1] X,
        const floating[:, ::1] Y,
        floating[:, ::1] D,
    ) except -1:
        """Compute the cross-pairwise distances between arrays X and Y"""
        cdef intp_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')
        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
        return 0

    cdef int pdist_csr(
        self,
        const floating* x1_data,
        const int32_t[::1] x1_indices,
        const int32_t[::1] x1_indptr,
        const intp_t size,
        floating[:, ::1] D,
    ) except -1 nogil:
        """Pairwise distances between rows in CSR matrix X.

        Note that this implementation is twice faster than cdist_csr(X, X)
        because it leverages the symmetry of the problem.
        """
        cdef:
            intp_t i1, i2
            intp_t n_x1 = x1_indptr.shape[0] - 1
            intp_t x1_start, x1_end, x2_start, x2_end

        for i1 in range(n_x1):
            x1_start = x1_indptr[i1]
            x1_end = x1_indptr[i1 + 1]
            for i2 in range(i1, n_x1):
                x2_start = x1_indptr[i2]
                x2_end = x1_indptr[i2 + 1]
                D[i1, i2] = D[i2, i1] = self.dist_csr(
                    x1_data,
                    &x1_indices[0],
                    x1_data,
                    &x1_indices[0],
                    x1_start,
                    x1_end,
                    x2_start,
                    x2_end,
                    size,
                )
        return 0

    cdef int cdist_csr(
        self,
        const floating* x1_data,
        const int32_t[::1] x1_indices,
        const int32_t[::1] x1_indptr,
        const floating* x2_data,
        const int32_t[::1] x2_indices,
        const int32_t[::1] x2_indptr,
        const intp_t size,
        floating[:, ::1] D,
    ) except -1 nogil:
        """Compute the cross-pairwise distances between arrays X and Y
        represented in the CSR format."""
        cdef:
            intp_t i1, i2
            intp_t n_x1 = x1_indptr.shape[0] - 1
            intp_t n_x2 = x2_indptr.shape[0] - 1
            intp_t x1_start, x1_end, x2_start, x2_end

        for i1 in range(n_x1):
            x1_start = x1_indptr[i1]
            x1_end = x1_indptr[i1 + 1]
            for i2 in range(n_x2):
                x2_start = x2_indptr[i2]
                x2_end = x2_indptr[i2 + 1]

                D[i1, i2] = self.dist_csr(
                    x1_data,
                    &x1_indices[0],
                    x2_data,
                    &x2_indices[0],
                    x1_start,
                    x1_end,
                    x2_start,
                    x2_end,
                    size,
                )
        return 0

    cdef float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        """Convert the rank-preserving surrogate distance to the distance"""
        if self.functor is NULL:
            return rdist
        if self._use_f32:
            return (<MetricBase[float32_t]*>self.functor).rdist_to_dist(<float32_t>rdist)
        return (<MetricBase[float64_t]*>self.functor).rdist_to_dist(rdist)

    cdef float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        """Convert the distance to the rank-preserving surrogate distance"""
        if self.functor is NULL:
            return dist
        if self._use_f32:
            return (<MetricBase[float32_t]*>self.functor).dist_to_rdist(<float32_t>dist)
        return (<MetricBase[float64_t]*>self.functor).dist_to_rdist(dist)

    def rdist_to_dist(self, rdist):
        """Convert the rank-preserving surrogate distance to the distance.

        The surrogate distance is any measure that yields the same rank as the
        distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Parameters
        ----------
        rdist : double
            Surrogate distance.

        Returns
        -------
        double
            True distance.
        """
        return rdist

    def dist_to_rdist(self, dist):
        """Convert the true distance to the rank-preserving surrogate distance.

        The surrogate distance is any measure that yields the same rank as the
        distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Parameters
        ----------
        dist : double
            True distance.

        Returns
        -------
        double
            Surrogate distance.
        """
        return dist

    def _pairwise_dense_dense(self, X, Y):
        cdef const float64_t[:, ::1] X64, Y64
        cdef const float32_t[:, ::1] X32, Y32
        cdef float64_t[:, ::1] D64
        cdef float32_t[:, ::1] D32

        if self._use_f32:
            X32 = np.asarray(X, dtype=np.float32, order='C')
            self._validate_data(X32)
            if X is Y:
                D32 = np.empty((X32.shape[0], X32.shape[0]), dtype=np.float32, order='C')
                self.pdist(X32, D32)
            else:
                Y32 = np.asarray(Y, dtype=np.float32, order='C')
                self._validate_data(Y32)
                D32 = np.empty((X32.shape[0], Y32.shape[0]), dtype=np.float32, order='C')
                self.cdist(X32, Y32, D32)
            return np.asarray(D32)

        X64 = np.asarray(X, dtype=np.float64, order='C')
        self._validate_data(X64)
        if X is Y:
            D64 = np.empty((X64.shape[0], X64.shape[0]), dtype=np.float64, order='C')
            self.pdist(X64, D64)
        else:
            Y64 = np.asarray(Y, dtype=np.float64, order='C')
            self._validate_data(Y64)
            D64 = np.empty((X64.shape[0], Y64.shape[0]), dtype=np.float64, order='C')
            self.cdist(X64, Y64, D64)
        return np.asarray(D64)

    def _pairwise_sparse_sparse(self, X: csr_matrix, Y: csr_matrix):
        cdef:
            intp_t n_X, n_features
            const float64_t[::1] X_data64, Y_data64
            const float32_t[::1] X_data32, Y_data32
            const int32_t[::1] X_indices, X_indptr, Y_indices, Y_indptr
            intp_t n_Y
            float64_t[:, ::1] D64
            float32_t[:, ::1] D32

        X_csr = X.tocsr()
        n_X, n_features = X_csr.shape
        X_indices = np.asarray(X_csr.indices, dtype=np.int32)
        X_indptr = np.asarray(X_csr.indptr, dtype=np.int32)
        if X is not Y:
            Y_csr = Y.tocsr()
            n_Y, _ = Y_csr.shape
            Y_indices = np.asarray(Y_csr.indices, dtype=np.int32)
            Y_indptr = np.asarray(Y_csr.indptr, dtype=np.int32)

        if self._use_f32:
            X_data32 = np.asarray(X_csr.data, dtype=np.float32)
            if X is Y:
                D32 = np.empty((n_X, n_X), dtype=np.float32, order='C')
                self.pdist_csr(&X_data32[0], X_indices, X_indptr, n_features, D32)
            else:
                Y_data32 = np.asarray(Y_csr.data, dtype=np.float32)
                D32 = np.empty((n_X, n_Y), dtype=np.float32, order='C')
                self.cdist_csr(&X_data32[0], X_indices, X_indptr,
                               &Y_data32[0], Y_indices, Y_indptr, n_features, D32)
            return np.asarray(D32)

        X_data64 = np.asarray(X_csr.data, dtype=np.float64)
        if X is Y:
            D64 = np.empty((n_X, n_X), dtype=np.float64, order='C')
            self.pdist_csr(&X_data64[0], X_indices, X_indptr, n_features, D64)
        else:
            Y_data64 = np.asarray(Y_csr.data, dtype=np.float64)
            D64 = np.empty((n_X, n_Y), dtype=np.float64, order='C')
            self.cdist_csr(&X_data64[0], X_indices, X_indptr,
                           &Y_data64[0], Y_indices, Y_indptr, n_features, D64)
        return np.asarray(D64)

    def _pairwise_sparse_dense(self, X: csr_matrix, Y):
        cdef:
            intp_t n_X = X.shape[0]
            intp_t n_features = X.shape[1]
            const int32_t[::1] X_indices = np.asarray(X.indices, dtype=np.int32)
            const int32_t[::1] X_indptr = np.asarray(X.indptr, dtype=np.int32)
            const int32_t[::1] Y_indices = np.arange(n_features, dtype=np.int32)
            const float64_t[::1] X_data64
            const float32_t[::1] X_data32
            const float64_t[:, ::1] Y_data64
            const float32_t[:, ::1] Y_data32
            float64_t[:, ::1] D64
            float32_t[:, ::1] D32
            intp_t n_Y, i1, i2, x1_start, x1_end

        # Use the exact same adaptation for CSR than in SparseDenseDatasetsPair
        # for supporting the sparse-dense case with minimal overhead.
        # Note: at this point this method is only a convenience method used in
        # the tests via the DistanceMetric.pairwise method. Efficient parallel
        # computation is achieved via the PairwiseDistances class instead.
        if self._use_f32:
            X_data32 = np.asarray(X.data, dtype=np.float32)
            Y_data32 = np.asarray(Y, dtype=np.float32, order="C")
            n_Y = Y_data32.shape[0]
            D32 = np.empty((n_X, n_Y), dtype=np.float32, order='C')
            with nogil:
                for i1 in range(n_X):
                    x1_start = X_indptr[i1]
                    x1_end = X_indptr[i1 + 1]
                    for i2 in range(n_Y):
                        D32[i1, i2] = self.dist_csr(
                            &X_data32[0], &X_indices[0],
                            &Y_data32[0, 0] + i2 * n_features, &Y_indices[0],
                            x1_start, x1_end, 0, n_features, n_features,
                        )
            return np.asarray(D32)

        X_data64 = np.asarray(X.data, dtype=np.float64)
        Y_data64 = np.asarray(Y, dtype=np.float64, order="C")
        n_Y = Y_data64.shape[0]
        D64 = np.empty((n_X, n_Y), dtype=np.float64, order='C')
        with nogil:
            for i1 in range(n_X):
                x1_start = X_indptr[i1]
                x1_end = X_indptr[i1 + 1]
                for i2 in range(n_Y):
                    D64[i1, i2] = self.dist_csr(
                        &X_data64[0], &X_indices[0],
                        &Y_data64[0, 0] + i2 * n_features, &Y_indices[0],
                        x1_start, x1_end, 0, n_features, n_features,
                    )
        return np.asarray(D64)

    def _pairwise_dense_sparse(self, X, Y: csr_matrix):
        cdef:
            intp_t n_X = X.shape[0]
            intp_t n_features = X.shape[1]
            const int32_t[::1] X_indices = np.arange(n_features, dtype=np.int32)
            intp_t n_Y = Y.shape[0]
            const int32_t[::1] Y_indices = np.asarray(Y.indices, dtype=np.int32)
            const int32_t[::1] Y_indptr = np.asarray(Y.indptr, dtype=np.int32)
            const float64_t[:, ::1] X_data64
            const float32_t[:, ::1] X_data32
            const float64_t[::1] Y_data64
            const float32_t[::1] Y_data32
            float64_t[:, ::1] D64
            float32_t[:, ::1] D32
            intp_t i1, i2, x2_start, x2_end

        if self._use_f32:
            X_data32 = np.asarray(X, dtype=np.float32, order="C")
            Y_data32 = np.asarray(Y.data, dtype=np.float32)
            D32 = np.empty((n_X, n_Y), dtype=np.float32, order='C')
            with nogil:
                for i1 in range(n_X):
                    for i2 in range(n_Y):
                        x2_start = Y_indptr[i2]
                        x2_end = Y_indptr[i2 + 1]
                        D32[i1, i2] = self.dist_csr(
                            &X_data32[0, 0] + i1 * n_features, &X_indices[0],
                            &Y_data32[0], &Y_indices[0],
                            0, n_features, x2_start, x2_end, n_features,
                        )
            return np.asarray(D32)

        X_data64 = np.asarray(X, dtype=np.float64, order="C")
        Y_data64 = np.asarray(Y.data, dtype=np.float64)
        D64 = np.empty((n_X, n_Y), dtype=np.float64, order='C')
        with nogil:
            for i1 in range(n_X):
                for i2 in range(n_Y):
                    x2_start = Y_indptr[i2]
                    x2_end = Y_indptr[i2 + 1]
                    D64[i1, i2] = self.dist_csr(
                        &X_data64[0, 0] + i1 * n_features, &X_indices[0],
                        &Y_data64[0], &Y_indices[0],
                        0, n_features, x2_start, x2_end, n_features,
                    )
        return np.asarray(D64)

    def pairwise(self, X, Y=None):
        """Compute the pairwise distances between X and Y

        This is a convenience routine for the sake of testing.  For many
        metrics, the utilities in scipy.spatial.distance.cdist and
        scipy.spatial.distance.pdist will be faster.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.
        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.
            If not specified, then Y=X.

        Returns
        -------
        dist : ndarray of shape  (n_samples_X, n_samples_Y)
            The distance matrix of pairwise distances between points in X and Y.
        """
        X = check_array(X, accept_sparse=['csr'])

        if Y is None:
            Y = X
        else:
            Y = check_array(Y, accept_sparse=['csr'])

        X_is_sparse = issparse(X)
        Y_is_sparse = issparse(Y)

        if not X_is_sparse and not Y_is_sparse:
            return self._pairwise_dense_dense(X, Y)

        if X_is_sparse and Y_is_sparse:
            return self._pairwise_sparse_sparse(X, Y)

        if X_is_sparse and not Y_is_sparse:
            return self._pairwise_sparse_dense(X, Y)

        return self._pairwise_dense_sparse(X, Y)


# ------------------------------------------------------------
# Euclidean Distance
#  d = sqrt(sum(x_i^2 - y_i^2))
cdef class EuclideanDistance(DistanceMetric):
    r"""Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }
    """
    def __init__(self):
        self.p = 2
        self._build_functor()

    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_euclidean[float32_t]()
        else:
            self.functor = <void*>make_euclidean[float64_t]()

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2


# ------------------------------------------------------------
# SEuclidean Distance
#  d = sqrt(sum((x_i - y_i2)^2 / v_i))
cdef class SEuclideanDistance(DistanceMetric):
    r"""Standardized Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i \frac{ (x_i - y_i) ^ 2}{V_i} }
    """
    def __init__(self, V):
        self.vec = np.asarray(V, dtype=np.float64)
        self.size = self.vec.shape[0]
        self.p = 2
        self._build_functor()

    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_seuclidean[float32_t](&self.vec[0], self.vec.shape[0])
        else:
            self.functor = <void*>make_seuclidean[float64_t](&self.vec[0], self.vec.shape[0])

    def _validate_data(self, X):
        if X.shape[1] != self.size:
            raise ValueError('SEuclidean dist: size of V does not match')

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2


# ------------------------------------------------------------
# Manhattan Distance
#  d = sum(abs(x_i - y_i))
cdef class ManhattanDistance(DistanceMetric):
    r"""Manhattan/City-block Distance metric

    .. math::
       D(x, y) = \sum_i |x_i - y_i|
    """
    def __init__(self):
        self.p = 1
        self._build_functor()

    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_manhattan[float32_t]()
        else:
            self.functor = <void*>make_manhattan[float64_t]()


# ------------------------------------------------------------
# Chebyshev Distance
#  d = max_i(abs(x_i - y_i))
cdef class ChebyshevDistance(DistanceMetric):
    """Chebyshev/Infinity Distance

    .. math::
       D(x, y) = max_i (|x_i - y_i|)

    Examples
    --------
    >>> from sklearn.metrics.dist_metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('chebyshev')
    >>> X = [[0, 1, 2],
    ...      [3, 4, 5]]
    >>> Y = [[-1, 0, 1],
    ...      [3, 4, 5]]
    >>> dist.pairwise(X, Y)
    array([[1.732..., 5.196...],
           [6.928..., 0....   ]])
    """
    def __init__(self):
        self.p = INF
        self._build_functor()

    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_chebyshev[float32_t]()
        else:
            self.functor = <void*>make_chebyshev[float64_t]()


# ------------------------------------------------------------
# Minkowski Distance
cdef class MinkowskiDistance(DistanceMetric):
    r"""Minkowski Distance

    .. math::
        D(x, y) = {||u-v||}_p

    when w is None.

    Here is the more general expanded expression for the weighted case:

    .. math::
        D(x, y) = [\sum_i w_i *|x_i - y_i|^p] ^ (1/p)

    Parameters
    ----------
    p : float
        The order of the p-norm of the difference (see above).

        .. versionchanged:: 1.4.0
            Minkowski distance allows `p` to be `0<p<1`.


    w : (N,) array-like (optional)
        The weight vector.

    Minkowski Distance requires p > 0 and finite.
    When :math:`p \in (0,1)`, it isn't a true metric but is permissible when
    the triangular inequality isn't necessary.
    For p = infinity, use ChebyshevDistance.
    Note that for p=1, ManhattanDistance is more efficient, and for
    p=2, EuclideanDistance is more efficient.

    """
    def __init__(self, p, w=None):
        if p <= 0:
            raise ValueError("p must be greater than 0")
        elif np.isinf(p):
            raise ValueError("MinkowskiDistance requires finite p. "
                             "For p=inf, use ChebyshevDistance.")

        self.p = p
        if w is not None:
            w_array = check_array(
                w, ensure_2d=False, dtype=np.float64, input_name="w"
            )
            if (w_array < 0).any():
                raise ValueError("w cannot contain negative weights")
            self.vec = w_array
            self.size = self.vec.shape[0]
        else:
            self.vec = np.asarray([], dtype=np.float64)
            self.size = 0
        self._build_functor()

    cdef void _build_functor(self) except *:
        cdef const double* w_ptr = NULL
        if self.vec.shape[0] > 0:
            w_ptr = &self.vec[0]
        if self._use_f32:
            self.functor = <void*>make_minkowski[float32_t](self.p, w_ptr, self.vec.shape[0])
        else:
            self.functor = <void*>make_minkowski[float64_t](self.p, w_ptr, self.vec.shape[0])

    def _validate_data(self, X):
        if self.size > 0 and X.shape[1] != self.size:
            raise ValueError("MinkowskiDistance: the size of w must match "
                             f"the number of features ({X.shape[1]}). "
                             f"Currently len(w)={self.size}.")

    def rdist_to_dist(self, rdist):
        return rdist ** (1. / self.p)

    def dist_to_rdist(self, dist):
        return dist ** self.p


# ------------------------------------------------------------
# Mahalanobis Distance
#  d = sqrt( (x - y)^T V^-1 (x - y) )
cdef class MahalanobisDistance(DistanceMetric):
    r"""Mahalanobis Distance

    .. math::
       D(x, y) = \sqrt{ (x - y)^T V^{-1} (x - y) }

    Parameters
    ----------
    V : array-like
        Symmetric positive-definite covariance matrix.
        The inverse of this matrix will be explicitly computed.
    VI : array-like
        optionally specify the inverse directly.  If VI is passed,
        then V is not referenced.
    """
    def __init__(self, V=None, VI=None):
        if VI is None:
            if V is None:
                raise ValueError("Must provide either V or VI "
                                 "for Mahalanobis distance")
            VI = np.linalg.inv(V)
        if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
            raise ValueError("V/VI must be square")

        self.mat = np.asarray(VI, dtype=np.float64, order='C')

        self.size = self.mat.shape[0]
        self._build_functor()

    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_mahalanobis[float32_t](&self.mat[0, 0], self.mat.shape[0])
        else:
            self.functor = <void*>make_mahalanobis[float64_t](&self.mat[0, 0], self.mat.shape[0])

    def __setstate__(self, state):
        DistanceMetric.__setstate__(self, state)
        # `mat` (VI), not `vec`, determines the feature count for Mahalanobis.
        self.size = self.mat.shape[0]
        self._free_functor()
        self._build_functor()

    def _validate_data(self, X):
        if X.shape[1] != self.size:
            raise ValueError('Mahalanobis dist: size of V does not match')

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2


# ------------------------------------------------------------
# Hamming Distance
#  d = N_unequal(x, y) / N_tot
cdef class HammingDistance(DistanceMetric):
    r"""Hamming Distance

    Hamming distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{1}{N} \sum_i \delta_{x_i, y_i}
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_hamming[float32_t]()
        else:
            self.functor = <void*>make_hamming[float64_t]()


# ------------------------------------------------------------
# Canberra Distance
#  D(x, y) = sum[ abs(x_i - y_i) / (abs(x_i) + abs(y_i)) ]
cdef class CanberraDistance(DistanceMetric):
    r"""Canberra Distance

    Canberra distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_canberra[float32_t]()
        else:
            self.functor = <void*>make_canberra[float64_t]()


# ------------------------------------------------------------
# Bray-Curtis Distance
#  D(x, y) = sum[abs(x_i - y_i)] / sum[abs(x_i) + abs(y_i)]
cdef class BrayCurtisDistance(DistanceMetric):
    r"""Bray-Curtis Distance

    Bray-Curtis distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i(|x_i| + |y_i|)}
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_braycurtis[float32_t]()
        else:
            self.functor = <void*>make_braycurtis[float64_t]()


# ------------------------------------------------------------
# Jaccard Distance (boolean)
#  D(x, y) = N_unequal(x, y) / N_nonzero(x, y)
cdef class JaccardDistance(DistanceMetric):
    r"""Jaccard Distance

    Jaccard Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (N_TT + N_TF + N_FT)
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_jaccard[float32_t]()
        else:
            self.functor = <void*>make_jaccard[float64_t]()


# ------------------------------------------------------------
# Matching Distance (boolean)
#  D(x, y) = n_neq / n
cdef class MatchingDistance(DistanceMetric):
    r"""Matching Distance

    Matching Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / N
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_matching[float32_t]()
        else:
            self.functor = <void*>make_matching[float64_t]()


# ------------------------------------------------------------
# Dice Distance (boolean)
#  D(x, y) = n_neq / (2 * ntt + n_neq)
cdef class DiceDistance(DistanceMetric):
    r"""Dice Distance

    Dice Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (2 * N_TT + N_TF + N_FT)

    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_dice[float32_t]()
        else:
            self.functor = <void*>make_dice[float64_t]()


# ------------------------------------------------------------
# Kulsinski Distance (boolean)
#  D(x, y) = (ntf + nft - ntt + n) / (n_neq + n)
cdef class KulsinskiDistance(DistanceMetric):
    r"""Kulsinski Distance

    Kulsinski Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 1 - N_TT / (N + N_TF + N_FT)

    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_kulsinski[float32_t]()
        else:
            self.functor = <void*>make_kulsinski[float64_t]()


# ------------------------------------------------------------
# Rogers-Tanimoto Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class RogersTanimotoDistance(DistanceMetric):
    r"""Rogers-Tanimoto Distance

    Rogers-Tanimoto Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 2 (N_TF + N_FT) / (N + N_TF + N_FT)
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_rogerstanimoto[float32_t]()
        else:
            self.functor = <void*>make_rogerstanimoto[float64_t]()


# ------------------------------------------------------------
# Russell-Rao Distance (boolean)
#  D(x, y) = (n - ntt) / n
cdef class RussellRaoDistance(DistanceMetric):
    r"""Russell-Rao Distance

    Russell-Rao Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N - N_TT) / N
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_russellrao[float32_t]()
        else:
            self.functor = <void*>make_russellrao[float64_t]()


# ------------------------------------------------------------
# Sokal-Michener Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class SokalMichenerDistance(DistanceMetric):
    r"""Sokal-Michener Distance

    Sokal-Michener Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 2 (N_TF + N_FT) / (N + N_TF + N_FT)
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_sokalmichener[float32_t]()
        else:
            self.functor = <void*>make_sokalmichener[float64_t]()


# ------------------------------------------------------------
# Sokal-Sneath Distance (boolean)
#  D(x, y) = n_neq / (0.5 * n_tt + n_neq)
cdef class SokalSneathDistance(DistanceMetric):
    r"""Sokal-Sneath Distance

    Sokal-Sneath Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (N_TT / 2 + N_FT + N_TF)
    """
    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_sokalsneath[float32_t]()
        else:
            self.functor = <void*>make_sokalsneath[float64_t]()


# ------------------------------------------------------------
# Haversine Distance (2 dimensional)
#  D(x, y) = 2 arcsin{sqrt[sin^2 ((x1 - y1) / 2)
#                          + cos(x1) cos(y1) sin^2 ((x2 - y2) / 2)]}
cdef class HaversineDistance(DistanceMetric):
    """Haversine (Spherical) Distance

    The Haversine distance is the angular distance between two points on
    the surface of a sphere.  The first distance of each point is assumed
    to be the latitude, the second is the longitude, given in radians.
    The dimension of the points must be 2:

    D(x, y) = 2 arcsin[sqrt{sin^2((x1 - y1) / 2) + cos(x1)cos(y1)sin^2((x2 - y2) / 2)}]

    """

    def _validate_data(self, X):
        if X.shape[1] != 2:
            raise ValueError("Haversine distance only valid "
                             "in 2 dimensions")

    cdef void _build_functor(self) except *:
        if self._use_f32:
            self.functor = <void*>make_haversine[float32_t]()
        else:
            self.functor = <void*>make_haversine[float64_t]()

    def rdist_to_dist(self, rdist):
        return 2 * np.arcsin(np.sqrt(rdist))

    def dist_to_rdist(self, dist):
        tmp = np.sin(0.5 * dist)
        return tmp * tmp


# ------------------------------------------------------------
# User-defined distance
#
cdef class PyFuncDistance(DistanceMetric):
    """PyFunc Distance

    A user-defined distance

    Parameters
    ----------
    func : function
        func should take two numpy arrays as input, and return a distance.
    """
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        # PyFunc keeps ``functor`` NULL; the base dist/rdist methods detect this
        # and call back into Python (with the GIL) via ``_pyfunc_dist``.

    cdef void _build_functor(self) except *:
        # No C++ functor: the Python callable is invoked directly.
        self.functor = NULL


######################################################################
# metric mappings
#  This maps from metric id strings to class names
METRIC_MAPPING = {
    'euclidean': EuclideanDistance,
    'l2': EuclideanDistance,
    'minkowski': MinkowskiDistance,
    'p': MinkowskiDistance,
    'manhattan': ManhattanDistance,
    'cityblock': ManhattanDistance,
    'l1': ManhattanDistance,
    'chebyshev': ChebyshevDistance,
    'infinity': ChebyshevDistance,
    'seuclidean': SEuclideanDistance,
    'mahalanobis': MahalanobisDistance,
    'hamming': HammingDistance,
    'canberra': CanberraDistance,
    'braycurtis': BrayCurtisDistance,
    'matching': MatchingDistance,
    'jaccard': JaccardDistance,
    'dice': DiceDistance,
    'kulsinski': KulsinskiDistance,
    'rogerstanimoto': RogersTanimotoDistance,
    'russellrao': RussellRaoDistance,
    'sokalmichener': SokalMichenerDistance,
    'sokalsneath': SokalSneathDistance,
    'haversine': HaversineDistance,
    'pyfunc': PyFuncDistance,
}


######################################################################
# Backward-compatibility aliases
#
#  Before this module was de-Tempita'd, the dtype was baked into the class
#  (``DistanceMetric64`` / ``EuclideanDistance32`` / ...). The dtype is now a
#  runtime attribute of a single class, but the historical names are kept as
#  plain Python aliases so that downstream imports, ``__class__.__name__``
#  look-ups via ``get_valid_metric_ids``, and old pickles keep working.
DistanceMetric64 = DistanceMetric
DistanceMetric32 = DistanceMetric
EuclideanDistance64 = EuclideanDistance
EuclideanDistance32 = EuclideanDistance
SEuclideanDistance64 = SEuclideanDistance
SEuclideanDistance32 = SEuclideanDistance
ManhattanDistance64 = ManhattanDistance
ManhattanDistance32 = ManhattanDistance
ChebyshevDistance64 = ChebyshevDistance
ChebyshevDistance32 = ChebyshevDistance
MinkowskiDistance64 = MinkowskiDistance
MinkowskiDistance32 = MinkowskiDistance
MahalanobisDistance64 = MahalanobisDistance
MahalanobisDistance32 = MahalanobisDistance
HammingDistance64 = HammingDistance
HammingDistance32 = HammingDistance
CanberraDistance64 = CanberraDistance
CanberraDistance32 = CanberraDistance
BrayCurtisDistance64 = BrayCurtisDistance
BrayCurtisDistance32 = BrayCurtisDistance
JaccardDistance64 = JaccardDistance
JaccardDistance32 = JaccardDistance
MatchingDistance64 = MatchingDistance
MatchingDistance32 = MatchingDistance
DiceDistance64 = DiceDistance
DiceDistance32 = DiceDistance
KulsinskiDistance64 = KulsinskiDistance
KulsinskiDistance32 = KulsinskiDistance
RogersTanimotoDistance64 = RogersTanimotoDistance
RogersTanimotoDistance32 = RogersTanimotoDistance
RussellRaoDistance64 = RussellRaoDistance
RussellRaoDistance32 = RussellRaoDistance
SokalMichenerDistance64 = SokalMichenerDistance
SokalMichenerDistance32 = SokalMichenerDistance
SokalSneathDistance64 = SokalSneathDistance
SokalSneathDistance32 = SokalSneathDistance
HaversineDistance64 = HaversineDistance
HaversineDistance32 = HaversineDistance
PyFuncDistance64 = PyFuncDistance
PyFuncDistance32 = PyFuncDistance

# Historical dtype-specialized metric-id maps (kept for backward compatibility).
METRIC_MAPPING64 = METRIC_MAPPING
METRIC_MAPPING32 = METRIC_MAPPING
