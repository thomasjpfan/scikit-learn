from cython cimport floating
from libc.math cimport sqrt

from sklearn.utils._typedefs cimport float64_t, float32_t, int32_t, intp_t


######################################################################
# Inline euclidean distance helpers
#
#  These exist for the default (euclidean) case so that they can be inlined
#  into hot loops (e.g. in the BallTree/KDTree), avoiding a virtual call.
#  They accumulate and return in float64, matching the previous
#  Tempita-generated ``euclidean_*{32,64}`` helpers. The dtype-suffixed names
#  are kept because ``sklearn/neighbors/_binary_tree.pxi.tp`` cimports them by
#  name. The loops are written out per dtype (rather than fused) to keep this
#  header simple for the many modules that cimport it.
cdef inline float64_t euclidean_rdist64(
    const float64_t* x1, const float64_t* x2, intp_t size,
) except -1 nogil:
    cdef float64_t tmp, d=0
    cdef intp_t j
    for j in range(size):
        tmp = <float64_t>(x1[j] - x2[j])
        d += tmp * tmp
    return d


cdef inline float64_t euclidean_rdist32(
    const float32_t* x1, const float32_t* x2, intp_t size,
) except -1 nogil:
    cdef float64_t tmp, d=0
    cdef intp_t j
    for j in range(size):
        tmp = <float64_t>(x1[j] - x2[j])
        d += tmp * tmp
    return d


cdef inline float64_t euclidean_dist64(
    const float64_t* x1, const float64_t* x2, intp_t size,
) except -1 nogil:
    return sqrt(euclidean_rdist64(x1, x2, size))


cdef inline float64_t euclidean_dist32(
    const float32_t* x1, const float32_t* x2, intp_t size,
) except -1 nogil:
    return sqrt(euclidean_rdist32(x1, x2, size))


cdef inline float64_t euclidean_dist_to_rdist64(const float64_t dist) except -1 nogil:
    return dist * dist


cdef inline float64_t euclidean_dist_to_rdist32(const float32_t dist) except -1 nogil:
    return dist * dist


cdef inline float64_t euclidean_rdist_to_dist64(const float64_t dist) except -1 nogil:
    return sqrt(dist)


cdef inline float64_t euclidean_rdist_to_dist32(const float32_t dist) except -1 nogil:
    return sqrt(dist)


######################################################################
# DistanceMetric base class
#
#  A single, dtype-agnostic ``DistanceMetric``. The per-vector distance math
#  lives once, in C++ (sklearn/metrics/src/metric_kernels.hpp), behind a
#  ``MetricBase<T>`` functor that this class owns (``functor``) and builds for
#  the configured dtype (``_use_f32``). The cdef ``dist``/``rdist``/... methods
#  fuse over the input pointer dtype (``floating``) but always return float64,
#  so callers can request float32 or float64 inputs from a single object.
cdef class DistanceMetric:
    # The following attributes are required for a few of the subclasses.
    cdef float64_t p
    cdef const float64_t[::1] vec
    cdef const float64_t[:, ::1] mat
    cdef intp_t size
    cdef object func
    cdef object kwargs

    # ``functor`` points to a ``MetricBase<float>`` (when ``_use_f32``) or a
    # ``MetricBase<double>``; it is NULL for PyFunc (handled in Python instead).
    cdef void* functor
    cdef bint _use_f32

    cdef float64_t dist(
        self,
        const floating* x1,
        const floating* x2,
        intp_t size,
    ) except -1 nogil

    cdef float64_t rdist(
        self,
        const floating* x1,
        const floating* x2,
        intp_t size,
    ) except -1 nogil

    # PyFunc fallback: invoked (with the GIL) when ``functor`` is NULL.
    cdef float64_t _pyfunc_dist(
        self,
        const floating* x1,
        const floating* x2,
        intp_t size,
    ) except -1 with gil

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
    ) except -1 nogil

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
    ) except -1 nogil

    cdef int pdist(
        self,
        const floating[:, ::1] X,
        floating[:, ::1] D,
    ) except -1

    cdef int cdist(
        self,
        const floating[:, ::1] X,
        const floating[:, ::1] Y,
        floating[:, ::1] D,
    ) except -1

    cdef int pdist_csr(
        self,
        const floating* x1_data,
        const int32_t[::1] x1_indices,
        const int32_t[::1] x1_indptr,
        const intp_t size,
        floating[:, ::1] D,
    ) except -1 nogil

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
    ) except -1 nogil

    cdef float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil

    cdef float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil

    # Build ``functor`` from the instance parameters for the configured dtype.
    cdef void _build_functor(self) except *

    # Delete ``functor`` (using ``_use_f32`` to pick the static type).
    cdef void _free_functor(self) noexcept
