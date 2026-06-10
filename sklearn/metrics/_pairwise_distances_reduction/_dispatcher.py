# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from abc import abstractmethod
from numbers import Integral, Real
from typing import List

import numpy as np
from scipy.sparse import issparse

from sklearn import get_config
from sklearn.metrics._dist_metrics import BOOL_METRICS, METRIC_MAPPING64, DistanceMetric
from sklearn.metrics._pairwise_distances_reduction._argkmin import ArgKmin32, ArgKmin64
from sklearn.metrics._pairwise_distances_reduction._argkmin_classmode import (
    ArgKminClassMode32,
    ArgKminClassMode64,
)
from sklearn.metrics._pairwise_distances_reduction._base import (
    _sqeuclidean_row_norms32,
    _sqeuclidean_row_norms64,
)
from sklearn.metrics._pairwise_distances_reduction._radius_neighbors import (
    RadiusNeighbors32,
    RadiusNeighbors64,
)
from sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode import (
    RadiusNeighborsClassMode32,
    RadiusNeighborsClassMode64,
)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.fixes import _in_unstable_openblas_configuration
from sklearn.utils.parallel import _get_threadpool_controller

# --- C++/nanobind backend selection ------------------------------------------
# The C++ reductions are the default. The legacy Cython implementation is still
# available as a fallback for the cases the C++ backend does not (yet) handle,
# and can be forced for the whole module by setting
# SKLEARN_PAIRWISE_DIST_BACKEND=cython. This override is temporary and will be
# removed together with the Cython implementation.
_STRATEGY_TO_INT = {"auto": 0, "parallel_on_X": 1, "parallel_on_Y": 2}

_CPP_GEMM_AVAILABLE = None


def _cpp_backend_enabled():
    return os.environ.get("SKLEARN_PAIRWISE_DIST_BACKEND", "cpp").lower() != "cython"


def _cpp_gemm_available():
    # The Euclidean GEMM specialization needs an LP64 (32-bit int) BLAS; the C++
    # module detects this from scipy's cython_blas capsule. Cached after first use.
    global _CPP_GEMM_AVAILABLE
    if _CPP_GEMM_AVAILABLE is None:
        from sklearn.metrics._pairwise_distances_reduction import _reductions

        _CPP_GEMM_AVAILABLE = bool(_reductions.gemm_available())
    return _CPP_GEMM_AVAILABLE


def _resolve_chunk_size(chunk_size):
    if chunk_size is None:
        chunk_size = get_config().get("pairwise_dist_chunk_size", 256)
    return int(chunk_size)


def _resolve_strategy(strategy):
    if strategy is None:
        strategy = get_config().get("pairwise_dist_parallel_strategy", "auto")
    return _STRATEGY_TO_INT[strategy]


def _cpp_is_dense(A):
    return (
        not issparse(A)
        and getattr(A, "ndim", None) == 2
        and getattr(getattr(A, "flags", None), "c_contiguous", False)
    )


def _cpp_is_csr(A):
    return (
        issparse(A)
        and A.format == "csr"
        and A.nnz > 0
        and A.indices.dtype == A.indptr.dtype == np.int32
    )


def _cpp_unpack_csr(A, dtype):
    return (
        np.ascontiguousarray(A.data, dtype=dtype),
        np.ascontiguousarray(A.indices, dtype=np.int32),
        np.ascontiguousarray(A.indptr, dtype=np.int32),
    )


def _cpp_ragged(flat, indptr):
    """Build a (n,) object array of per-row views from a flat C++ result."""
    out = np.empty(indptr.shape[0] - 1, dtype=object)
    for i in range(out.shape[0]):
        out[i] = flat[indptr[i] : indptr[i + 1]]
    return out


def _cpp_argkmin_neighbors(X, Y, k, addr, chunk_size, n_threads, strategy):
    """Run the C++ ArgKmin, returning (surrogate distances, indices) arrays.

    The distances are kept rank-preserving (use_squared_distances=True skips the
    exact-distance conversion), as the classmode weighted histogram expects.
    """
    from sklearn.metrics._pairwise_distances_reduction import _reductions

    args = (int(k), chunk_size, n_threads, strategy, addr, True, True)
    X_is_sparse, Y_is_sparse = issparse(X), issparse(Y)
    if not X_is_sparse and not Y_is_sparse:
        return _reductions.argkmin_dense_dense(X, Y, *args)
    if X_is_sparse and Y_is_sparse:
        Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
        Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
        return _reductions.argkmin_sparse_sparse(
            Xd, Xi, Xp, Yd, Yi, Yp, X.shape[1], *args
        )
    if X_is_sparse:
        Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
        return _reductions.argkmin_sparse_dense(Xd, Xi, Xp, Y, *args)
    Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
    return _reductions.argkmin_dense_sparse(X, Yd, Yi, Yp, *args)


def _cpp_radius_neighbors_flat(X, Y, r_radius, addr, chunk_size, n_threads, strategy):
    """Run the C++ RadiusNeighbors, returning flat (distances, indices, indptr).

    Distances are kept rank-preserving (convert_distances=False) for the
    classmode weighted histogram. sort_results is irrelevant for the histogram.
    """
    from sklearn.metrics._pairwise_distances_reduction import _reductions

    # (r_radius, sort_results, chunk, n_threads, strategy, addr,
    #  return_distance, convert_distances)
    args = (r_radius, False, chunk_size, n_threads, strategy, addr, True, False)
    X_is_sparse, Y_is_sparse = issparse(X), issparse(Y)
    if not X_is_sparse and not Y_is_sparse:
        return _reductions.radius_dense_dense(X, Y, *args)
    if X_is_sparse and Y_is_sparse:
        Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
        Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
        return _reductions.radius_sparse_sparse(
            Xd, Xi, Xp, Yd, Yi, Yp, X.shape[1], *args
        )
    if X_is_sparse:
        Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
        return _reductions.radius_sparse_dense(Xd, Xi, Xp, Y, *args)
    Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
    return _reductions.radius_dense_sparse(X, Yd, Yi, Yp, *args)


def _cpp_portable_instance(metric):
    """Whether a DistanceMetric *instance* can be run by the C++ reductions.

    PyFunc has no C++ functor; Mahalanobis is functor-backed but its functor has
    a non-reentrant scratch buffer (unsafe under the reductions' parallelism).
    """
    return hasattr(metric, "_functor_address") and (
        "Mahalanobis" not in type(metric).__name__
    )


def _cpp_metric_supported(cls, metric):
    if isinstance(metric, str):
        return metric in cls.valid_metrics()
    if isinstance(metric, DistanceMetric):
        return _cpp_portable_instance(metric)
    return False


def _cpp_resolve_metric(metric, dtype, metric_kwargs):
    """Return (functor-backed DistanceMetric, use_squared_distances).

    A DistanceMetric instance is used directly (its functor is borrowed); a
    string is built via get_metric (the single parser). "sqeuclidean" maps to the
    Euclidean functor with use_squared_distances=True.
    """
    if isinstance(metric, DistanceMetric):
        return metric, False
    use_squared_distances = metric == "sqeuclidean"
    name = "euclidean" if use_squared_distances else metric
    forwarded = {
        key: value
        for key, value in (metric_kwargs or {}).items()
        if key not in ("X_norm_squared", "Y_norm_squared")
    }
    return DistanceMetric.get_metric(
        name, dtype=dtype, **forwarded
    ), use_squared_distances


def _cpp_classmode_supported(cls, X, Y, metric, weights):
    return (
        _cpp_backend_enabled()
        and _cpp_metric_supported(cls, metric)
        and (_cpp_is_dense(X) or _cpp_is_csr(X))
        and (_cpp_is_dense(Y) or _cpp_is_csr(Y))
        and X.dtype == Y.dtype
        and X.dtype in (np.float32, np.float64)
        and weights in ("uniform", "distance")
    )


def sqeuclidean_row_norms(X, num_threads):
    """Compute the squared euclidean norm of the rows of X in parallel.

    Parameters
    ----------
    X : ndarray or CSR matrix of shape (n_samples, n_features)
        Input data. Must be c-contiguous.

    num_threads : int
        The number of OpenMP threads to use.

    Returns
    -------
    sqeuclidean_row_norms : ndarray of shape (n_samples,)
        Arrays containing the squared euclidean norm of each row of X.
    """
    if X.dtype == np.float64:
        return np.asarray(_sqeuclidean_row_norms64(X, num_threads))
    if X.dtype == np.float32:
        return np.asarray(_sqeuclidean_row_norms32(X, num_threads))

    raise ValueError(
        "Only float64 or float32 datasets are supported at this time, "
        f"got: X.dtype={X.dtype}."
    )


class BaseDistancesReductionDispatcher:
    """Abstract base dispatcher for pairwise distance computation & reduction.

    Each dispatcher extending the base :class:`BaseDistancesReductionDispatcher`
    dispatcher must implement the :meth:`compute` classmethod.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        excluded = {
            # PyFunc cannot be supported because it necessitates interacting with
            # the CPython interpreter to call user defined functions.
            "pyfunc",
            "mahalanobis",  # is numerically unstable
            # In order to support discrete distance metrics, we need to have a
            # stable simultaneous sort which preserves the order of the indices
            # because there generally is a lot of occurrences for a given values
            # of distances in this case.
            # TODO: implement a stable simultaneous_sort.
            "hamming",
            *BOOL_METRICS,
        }
        return sorted(({"sqeuclidean"} | set(METRIC_MAPPING64.keys())) - excluded)

    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool:
        """Return True if the dispatcher can be used for the
        given parameters.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.

        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)
            Input data.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        Returns
        -------
        True if the dispatcher can be used, else False.
        """

        # FIXME: the current Cython implementation is too slow for a large number of
        # features. We temporarily disable it to fallback on SciPy's implementation.
        # See: https://github.com/scikit-learn/scikit-learn/issues/28191
        if (
            issparse(X)
            and issparse(Y)
            and isinstance(metric, str)
            and "euclidean" in metric
        ):
            return False

        def is_numpy_c_ordered(X):
            return hasattr(X, "flags") and getattr(X.flags, "c_contiguous", False)

        def is_valid_sparse_matrix(X):
            return (
                issparse(X)
                and X.format == "csr"
                and
                # TODO: support CSR matrices without non-zeros elements
                X.nnz > 0
                and
                # TODO: support CSR matrices with int64 indices and indptr
                # See: https://github.com/scikit-learn/scikit-learn/issues/23653
                X.indices.dtype == X.indptr.dtype == np.int32
            )

        is_usable = (
            get_config().get("enable_cython_pairwise_dist", True)
            and (is_numpy_c_ordered(X) or is_valid_sparse_matrix(X))
            and (is_numpy_c_ordered(Y) or is_valid_sparse_matrix(Y))
            and X.dtype == Y.dtype
            and X.dtype in (np.float32, np.float64)
            and (
                metric in cls.valid_metrics()
                # DistanceMetric instances are usable except the ones the backend
                # cannot run (PyFunc has no C++ functor; Mahalanobis is not
                # reentrant under the parallel reductions).
                or (
                    isinstance(metric, DistanceMetric)
                    and _cpp_portable_instance(metric)
                )
            )
        )

        return is_usable

    @classmethod
    @abstractmethod
    def compute(
        cls,
        X,
        Y,
        **kwargs,
    ):
        """Compute the reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        **kwargs : additional parameters for the reduction

        Notes
        -----
        This method is an abstract class method: it has to be implemented
        for all subclasses.
        """


class ArgKmin(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances.

    ArgKmin is typically used to perform
    bruteforce k-nearest neighbors queries.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
    ):
        """Compute the argkmin reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        k : int
            The k for the argkmin reduction.

        metric : str, default='euclidean'
            The distance metric to use for argkmin.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        return_distance : boolean, default=False
            Return distances between each X vector and its
            argkmin if set to True.

        Returns
        -------
        If return_distance=False:
          - argkmin_indices : ndarray of shape (n_samples_X, k)
            Indices of the argkmin for each vector in X.

        If return_distance=True:
          - argkmin_distances : ndarray of shape (n_samples_X, k)
            Distances to the argkmin for each vector in X.
          - argkmin_indices : ndarray of shape (n_samples_X, k)
            Indices of the argkmin for each vector in X.

        Notes
        -----
        This classmethod inspects the arguments values to dispatch to the
        dtype-specialized implementation of :class:`ArgKmin`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """

        # Experimental C++/nanobind backend (private toggle). Routes valid,
        # supported dense cases to the C++ reductions; everything else
        # (validation, warnings, unsupported metrics, sparse) stays on the
        # Cython path. The generic metric is built by the single (Cython) parser
        # and the C++ side borrows its functor.
        def _cpp_argkmin_supported():
            if not (
                _cpp_metric_supported(cls, metric)
                and (_cpp_is_dense(X) or _cpp_is_csr(X))
                and (_cpp_is_dense(Y) or _cpp_is_csr(Y))
                and X.dtype == Y.dtype
                and X.dtype in (np.float32, np.float64)
                and isinstance(k, Integral)
                and k >= 1
            ):
                return False
            # For (sq)euclidean, extra metric_kwargs are ignored with a
            # UserWarning by the Cython path; defer those so the warning is kept.
            if metric in ("euclidean", "sqeuclidean") and metric_kwargs is not None:
                return set(metric_kwargs).issubset({"X_norm_squared", "Y_norm_squared"})
            return True

        if _cpp_backend_enabled() and _cpp_argkmin_supported():
            from sklearn.metrics._pairwise_distances_reduction import _reductions

            # Build (and keep alive for the whole call) the functor-backed metric.
            distance_metric, use_squared_distances = _cpp_resolve_metric(
                metric, X.dtype, metric_kwargs
            )
            # Metric-specific input checks, matching DatasetsPair.get_for.
            distance_metric._validate_data(X)
            distance_metric._validate_data(Y)

            common = (
                int(k),
                _resolve_chunk_size(chunk_size),
                _openmp_effective_n_threads(),
                _resolve_strategy(strategy),
                distance_metric._functor_address(),
                use_squared_distances,
                return_distance,
            )
            X_is_sparse, Y_is_sparse = issparse(X), issparse(Y)
            if not X_is_sparse and not Y_is_sparse:
                if (
                    metric in ("euclidean", "sqeuclidean")
                    and _cpp_gemm_available()
                    and not _in_unstable_openblas_configuration()
                ):
                    # GEMM specialization: limit BLAS to 1 thread to avoid
                    # over-subscription with the chunk-level OpenMP parallelism.
                    with _get_threadpool_controller().limit(limits=1, user_api="blas"):
                        return _reductions.euclidean_argkmin_dense_dense(X, Y, *common)
                return _reductions.argkmin_dense_dense(X, Y, *common)
            if X_is_sparse and Y_is_sparse:
                Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
                Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
                return _reductions.argkmin_sparse_sparse(
                    Xd, Xi, Xp, Yd, Yi, Yp, X.shape[1], *common
                )
            if X_is_sparse:
                Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
                return _reductions.argkmin_sparse_dense(Xd, Xi, Xp, Y, *common)
            Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
            return _reductions.argkmin_dense_sparse(X, Yd, Yi, Yp, *common)

        if X.dtype == Y.dtype == np.float64:
            return ArgKmin64.compute(
                X=X,
                Y=Y,
                k=k,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                return_distance=return_distance,
            )

        if X.dtype == Y.dtype == np.float32:
            return ArgKmin32.compute(
                X=X,
                Y=Y,
                k=k,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                return_distance=return_distance,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )


class RadiusNeighbors(BaseDistancesReductionDispatcher):
    """Compute radius-based neighbors for two sets of vectors.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    The distance function `dist` depends on the values of the `metric`
    and `metric_kwargs` parameters.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
        sort_results=False,
    ):
        """Return the results of the reduction for the given arguments.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        radius : float
            The radius defining the neighborhood.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        return_distance : boolean, default=False
            Return distances between each X vector and its neighbors if set to True.

        sort_results : boolean, default=False
            Sort results with respect to distances between each X vector and its
            neighbors if set to True.

        Returns
        -------
        If return_distance=False:
          - neighbors_indices : ndarray of n_samples_X ndarray
            Indices of the neighbors for each vector in X.

        If return_distance=True:
          - neighbors_indices : ndarray of n_samples_X ndarray
            Indices of the neighbors for each vector in X.
          - neighbors_distances : ndarray of n_samples_X ndarray
            Distances to the neighbors for each vector in X.

        Notes
        -----
        This classmethod inspects the arguments values to dispatch to the
        dtype-specialized implementation of :class:`RadiusNeighbors`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """

        def _cpp_radius_supported():
            if not (
                _cpp_metric_supported(cls, metric)
                and (_cpp_is_dense(X) or _cpp_is_csr(X))
                and (_cpp_is_dense(Y) or _cpp_is_csr(Y))
                and X.dtype == Y.dtype
                and X.dtype in (np.float32, np.float64)
                and isinstance(radius, Real)
                and radius >= 0
            ):
                return False
            if metric in ("euclidean", "sqeuclidean") and metric_kwargs is not None:
                return set(metric_kwargs).issubset({"X_norm_squared", "Y_norm_squared"})
            return True

        if _cpp_backend_enabled() and _cpp_radius_supported():
            from sklearn.metrics._pairwise_distances_reduction import _reductions

            distance_metric, use_squared_distances = _cpp_resolve_metric(
                metric, X.dtype, metric_kwargs
            )
            distance_metric._validate_data(X)
            distance_metric._validate_data(Y)

            # Rank-preserving threshold. For sqeuclidean the given radius is
            # already the squared radius.
            if use_squared_distances:
                r_radius = float(radius)
            else:
                r_radius = float(distance_metric.dist_to_rdist(radius))

            common = (
                r_radius,
                bool(sort_results),
                _resolve_chunk_size(chunk_size),
                _openmp_effective_n_threads(),
                _resolve_strategy(strategy),
                distance_metric._functor_address(),
                return_distance,
                True,  # convert_distances: return exact distances
            )
            X_is_sparse, Y_is_sparse = issparse(X), issparse(Y)
            if not X_is_sparse and not Y_is_sparse:
                res = _reductions.radius_dense_dense(X, Y, *common)
            elif X_is_sparse and Y_is_sparse:
                Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
                Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
                res = _reductions.radius_sparse_sparse(
                    Xd, Xi, Xp, Yd, Yi, Yp, X.shape[1], *common
                )
            elif X_is_sparse:
                Xd, Xi, Xp = _cpp_unpack_csr(X, X.dtype)
                res = _reductions.radius_sparse_dense(Xd, Xi, Xp, Y, *common)
            else:
                Yd, Yi, Yp = _cpp_unpack_csr(Y, Y.dtype)
                res = _reductions.radius_dense_sparse(X, Yd, Yi, Yp, *common)

            if return_distance:
                dist_flat, idx_flat, indptr = res
                return _cpp_ragged(dist_flat, indptr), _cpp_ragged(idx_flat, indptr)
            idx_flat, indptr = res
            return _cpp_ragged(idx_flat, indptr)

        if X.dtype == Y.dtype == np.float64:
            return RadiusNeighbors64.compute(
                X=X,
                Y=Y,
                radius=radius,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                sort_results=sort_results,
                return_distance=return_distance,
            )

        if X.dtype == Y.dtype == np.float32:
            return RadiusNeighbors32.compute(
                X=X,
                Y=Y,
                radius=radius,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                sort_results=sort_results,
                return_distance=return_distance,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )


class ArgKminClassMode(BaseDistancesReductionDispatcher):
    """Compute the argkmin of row vectors of X on the ones of Y with labels.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances. Computes weighted mode of labels.

    ArgKminClassMode is typically used to perform bruteforce k-nearest neighbors
    queries when the weighted mode of the labels for the k-nearest neighbors
    are required, such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        excluded = {
            # Euclidean is technically usable for ArgKminClassMode
            # but its current implementation would not be competitive.
            # TODO: implement Euclidean specialization using GEMM.
            "euclidean",
            "sqeuclidean",
        }
        return list(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        weights,
        Y_labels,
        unique_Y_labels,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
    ):
        """Compute the argkmin reduction.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            The input array to be labelled.

        Y : ndarray of shape (n_samples_Y, n_features)
            The input array whose class membership are provided through the
            `Y_labels` parameter.

        k : int
            The number of nearest neighbors to consider.

        weights : ndarray
            The weights applied over the `Y_labels` of `Y` when computing the
            weighted mode of the labels.

        Y_labels : ndarray
            An array containing the index of the class membership of the
            associated samples in `Y`. This is used in labeling `X`.

        unique_Y_labels : ndarray
            An array containing all unique indices contained in the
            corresponding `Y_labels` array.

        metric : str, default='euclidean'
            The distance metric to use. For a list of available metrics, see
            the documentation of :class:`~sklearn.metrics.DistanceMetric`.
            Currently does not support `'precomputed'`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        Returns
        -------
        probabilities : ndarray of shape (n_samples_X, n_classes)
            An array containing the class probabilities for each sample.

        Notes
        -----
        This classmethod is responsible for introspecting the arguments
        values to dispatch to the most appropriate implementation of
        :class:`PairwiseDistancesArgKmin`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """
        if weights not in {"uniform", "distance"}:
            raise ValueError(
                "Only the 'uniform' or 'distance' weights options are supported"
                f" at this time. Got: {weights=}."
            )

        # Experimental C++/nanobind backend (private toggle): run the C++ ArgKmin
        # and build the weighted label histogram (class probabilities) in numpy.
        if (
            _cpp_classmode_supported(cls, X, Y, metric, weights)
            and isinstance(k, Integral)
            and k >= 1
        ):
            Y_labels_arr = np.asarray(Y_labels, dtype=np.intp)
            n_classes = np.asarray(unique_Y_labels).shape[0]
            distance_metric, _ = _cpp_resolve_metric(metric, X.dtype, metric_kwargs)
            distance_metric._validate_data(X)
            distance_metric._validate_data(Y)
            distances, indices = _cpp_argkmin_neighbors(
                X,
                Y,
                k,
                distance_metric._functor_address(),
                _resolve_chunk_size(chunk_size),
                _openmp_effective_n_threads(),
                _resolve_strategy(strategy),
            )
            n_X = indices.shape[0]
            labels = Y_labels_arr[indices]  # (n_X, k)
            if weights == "uniform":
                scores = np.ones_like(distances)
            else:
                scores = 1.0 / distances
            bins = (np.arange(n_X)[:, None] * n_classes + labels).ravel()
            class_scores = np.bincount(
                bins, weights=scores.ravel(), minlength=n_X * n_classes
            ).reshape(n_X, n_classes)
            class_scores /= class_scores.sum(axis=1, keepdims=True)
            return class_scores

        if X.dtype == Y.dtype == np.float64:
            return ArgKminClassMode64.compute(
                X=X,
                Y=Y,
                k=k,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        if X.dtype == Y.dtype == np.float32:
            return ArgKminClassMode32.compute(
                X=X,
                Y=Y,
                k=k,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )


class RadiusNeighborsClassMode(BaseDistancesReductionDispatcher):
    """Compute radius-based class modes of row vectors of X using the
    those of Y.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    RadiusNeighborsClassMode is typically used to perform bruteforce
    radius neighbors queries when the weighted mode of the labels for
    the nearest neighbors within the specified radius are required,
    such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        excluded = {
            # Euclidean is technically usable for RadiusNeighborsClassMode
            # but it would not be competitive.
            # TODO: implement Euclidean specialization using GEMM.
            "euclidean",
            "sqeuclidean",
        }
        return sorted(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        weights,
        Y_labels,
        unique_Y_labels,
        outlier_label,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
    ):
        """Return the results of the reduction for the given arguments.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            The input array to be labelled.
        Y : ndarray of shape (n_samples_Y, n_features)
            The input array whose class membership is provided through
            the `Y_labels` parameter.
        radius : float
            The radius defining the neighborhood.
        weights : ndarray
            The weights applied to the `Y_labels` when computing the
            weighted mode of the labels.
        Y_labels : ndarray
            An array containing the index of the class membership of the
            associated samples in `Y`. This is used in labeling `X`.
        unique_Y_labels : ndarray
            An array containing all unique class labels.
        outlier_label : int, default=None
            Label for outlier samples (samples with no neighbors in given
            radius). In the default case when the value is None if any
            outlier is detected, a ValueError will be raised. The outlier
            label should be selected from among the unique 'Y' labels. If
            it is specified with a different value a warning will be raised
            and all class probabilities of outliers will be assigned to be 0.
        metric : str, default='euclidean'
            The distance metric to use. For a list of available metrics, see
            the documentation of :class:`~sklearn.metrics.DistanceMetric`.
            Currently does not support `'precomputed'`.
        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.
        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.
        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.
            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:
              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.
              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.
              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.
              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.
        Returns
        -------
        probabilities : ndarray of shape (n_samples_X, n_classes)
            An array containing the class probabilities for each sample.
        """
        if weights not in {"uniform", "distance"}:
            raise ValueError(
                "Only the 'uniform' or 'distance' weights options are supported"
                f" at this time. Got: {weights=}."
            )

        # Experimental C++/nanobind backend (private toggle): run the C++
        # RadiusNeighbors and build the weighted label histogram in numpy.
        if _cpp_classmode_supported(cls, X, Y, metric, weights) and (
            isinstance(radius, Real) and radius >= 0
        ):
            Y_labels_arr = np.asarray(Y_labels, dtype=np.intp)
            unique_Y_labels_arr = np.asarray(unique_Y_labels)
            n_classes = unique_Y_labels_arr.shape[0]
            distance_metric, _ = _cpp_resolve_metric(metric, X.dtype, metric_kwargs)
            distance_metric._validate_data(X)
            distance_metric._validate_data(Y)
            r_radius = float(distance_metric.dist_to_rdist(radius))
            dist_flat, idx_flat, indptr = _cpp_radius_neighbors_flat(
                X,
                Y,
                r_radius,
                distance_metric._functor_address(),
                _resolve_chunk_size(chunk_size),
                _openmp_effective_n_threads(),
                _resolve_strategy(strategy),
            )
            n_X = indptr.shape[0] - 1
            counts = np.diff(indptr)

            outlier_label_index = -1
            if outlier_label is not None:
                matches = np.flatnonzero(unique_Y_labels_arr == outlier_label)
                if matches.size:
                    outlier_label_index = int(matches[0])

            rows = np.repeat(np.arange(n_X), counts)
            labels = Y_labels_arr[idx_flat]
            if weights == "uniform":
                scores = np.ones(idx_flat.shape[0], dtype=np.float64)
            else:
                scores = 1.0 / dist_flat
            class_scores = np.bincount(
                rows * n_classes + labels, weights=scores, minlength=n_X * n_classes
            ).reshape(n_X, n_classes)

            outliers = counts == 0
            if outlier_label_index >= 0:
                class_scores[outliers, outlier_label_index] = 1.0
            if outliers.any() and outlier_label is None:
                raise ValueError(
                    "No neighbors found for test samples %r, "
                    "you can try using larger radius, "
                    "giving a label for outliers, "
                    "or considering removing them from your dataset."
                    % np.where(outliers)[0]
                )
            if outliers.any() and outlier_label_index < 0:
                warnings.warn(
                    "Outlier label %s is not in training "
                    "classes. All class probabilities of "
                    "outliers will be assigned with 0." % outlier_label
                )
            normalizer = class_scores.sum(axis=1, keepdims=True)
            normalizer[normalizer == 0.0] = 1.0
            class_scores /= normalizer
            return class_scores

        if X.dtype == Y.dtype == np.float64:
            return RadiusNeighborsClassMode64.compute(
                X=X,
                Y=Y,
                radius=radius,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                outlier_label=outlier_label,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        if X.dtype == Y.dtype == np.float32:
            return RadiusNeighborsClassMode32.compute(
                X=X,
                Y=Y,
                radius=radius,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                outlier_label=outlier_label,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )
