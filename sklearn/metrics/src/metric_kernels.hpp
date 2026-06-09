/* Single-source C++ implementation of scikit-learn's distance metric kernels.
 *
 * This header is the canonical implementation of the per-vector distance math
 * that used to live (duplicated for float32/float64 via Tempita) in
 * _dist_metrics.pyx.tp. It is consumed by:
 *   - sklearn/metrics/_dist_metrics.pyx.tp (the Cython DistanceMetric classes
 *     delegate their cdef dist/rdist/... methods to these functors), and
 *   - sklearn/metrics/_pairwise_distances_reduction (the C++/nanobind reductions
 *     borrow a functor pointer for the generic-metric path).
 *
 * Fidelity notes (these MUST be preserved to match the previous Cython output):
 *   - Virtual methods return `T` (float for float32 inputs), so the float32 path
 *     narrows the double-precision accumulation back to float exactly as the
 *     Cython code did by returning {{INPUT_DTYPE_t}}.
 *   - All libc-style math (sqrt/pow/fabs/sin/cos/asin) runs in double: operands
 *     are widened to double before the call, matching `from libc.math cimport`.
 *   - Accumulator types match the Cython source per-metric (e.g. Manhattan's CSR
 *     loop accumulates in `T`, while its dense loop accumulates in double).
 *   - Some CSR tail loops faithfully reproduce pre-existing quirks in the Cython
 *     implementation; these are called out with "FAITHFUL:" comments.
 *
 * PyFunc (user-callable) is intentionally NOT implemented here: it requires the
 * GIL and a Python call, and remains in Cython.
 */
#ifndef SKLEARN_METRICS_METRIC_KERNELS_HPP
#define SKLEARN_METRICS_METRIC_KERNELS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sklearn {
namespace metrics {

using std::asin;
using std::cos;
using std::fabs;
using std::pow;
using std::sin;
using std::sqrt;

using index_t = std::int32_t;   // CSR index dtype (int32, as enforced upstream)
using size_type = std::intptr_t;

// ---------------------------------------------------------------------------
// Abstract base
// ---------------------------------------------------------------------------
template <typename T>
struct MetricBase {
    virtual ~MetricBase() = default;

    // Dense interface. `dist` is the true distance; `rdist` is a rank-preserving
    // surrogate (cheaper, same ordering). By default rdist == dist.
    virtual T dist(const T* x1, const T* x2, size_type size) const = 0;
    virtual T rdist(const T* x1, const T* x2, size_type size) const {
        return dist(x1, x2, size);
    }

    // CSR interface; see _dist_metrics.pyx.tp `dist_csr` docstring for the
    // pointer + [start, end) index-pointer calling convention.
    virtual T dist_csr(const T* x1_data, const index_t* x1_indices,
                       const T* x2_data, const index_t* x2_indices,
                       index_t x1_start, index_t x1_end,
                       index_t x2_start, index_t x2_end,
                       size_type size) const = 0;
    virtual T rdist_csr(const T* x1_data, const index_t* x1_indices,
                        const T* x2_data, const index_t* x2_indices,
                        index_t x1_start, index_t x1_end,
                        index_t x2_start, index_t x2_end,
                        size_type size) const {
        return dist_csr(x1_data, x1_indices, x2_data, x2_indices,
                        x1_start, x1_end, x2_start, x2_end, size);
    }

    virtual T rdist_to_dist(T rdist) const { return rdist; }
    virtual T dist_to_rdist(T dist) const { return dist; }
};

// ---------------------------------------------------------------------------
// Euclidean:  d = sqrt(sum (x_i - y_i)^2)   rdist = sum (x_i - y_i)^2
// ---------------------------------------------------------------------------
template <typename T>
struct EuclideanDistance : MetricBase<T> {
    T rdist(const T* x1, const T* x2, size_type size) const override {
        double d = 0.0;
        for (size_type j = 0; j < size; ++j) {
            double tmp = x1[j] - x2[j];
            d += tmp * tmp;
        }
        return static_cast<T>(d);
    }
    T dist(const T* x1, const T* x2, size_type size) const override {
        return static_cast<T>(sqrt(static_cast<double>(rdist(x1, x2, size))));
    }
    T rdist_to_dist(T rdist) const override { return static_cast<T>(sqrt(static_cast<double>(rdist))); }
    T dist_to_rdist(T dist) const override { return static_cast<T>(static_cast<double>(dist) * dist); }

    T rdist_csr(const T* x1_data, const index_t* x1_indices,
                const T* x2_data, const index_t* x2_indices,
                index_t x1_start, index_t x1_end,
                index_t x2_start, index_t x2_end,
                size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double d = 0.0, unsquared = 0.0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) {
                unsquared = x1_data[i1] - x2_data[i2];
                d += unsquared * unsquared;
                ++i1; ++i2;
            } else if (ix1 < ix2) {
                unsquared = x1_data[i1]; d += unsquared * unsquared; ++i1;
            } else {
                unsquared = x2_data[i2]; d += unsquared * unsquared; ++i2;
            }
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) { unsquared = x2_data[i2]; d += unsquared * unsquared; ++i2; }
        } else {
            while (i1 < x1_end) { unsquared = x1_data[i1]; d += unsquared * unsquared; ++i1; }
        }
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        return static_cast<T>(sqrt(static_cast<double>(rdist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, size))));
    }
};

// ---------------------------------------------------------------------------
// Standardized Euclidean:  d = sqrt(sum (x_i - y_i)^2 / V_i)
// ---------------------------------------------------------------------------
template <typename T>
struct SEuclideanDistance : MetricBase<T> {
    std::vector<double> vec;  // V
    explicit SEuclideanDistance(const double* V, size_type n) : vec(V, V + n) {}

    T rdist(const T* x1, const T* x2, size_type size) const override {
        double d = 0.0;
        for (size_type j = 0; j < size; ++j) {
            double tmp = x1[j] - x2[j];
            d += tmp * tmp / vec[j];
        }
        return static_cast<T>(d);
    }
    T dist(const T* x1, const T* x2, size_type size) const override {
        return static_cast<T>(sqrt(static_cast<double>(rdist(x1, x2, size))));
    }
    T rdist_to_dist(T rdist) const override { return static_cast<T>(sqrt(static_cast<double>(rdist))); }
    T dist_to_rdist(T dist) const override { return static_cast<T>(static_cast<double>(dist) * dist); }

    T rdist_csr(const T* x1_data, const index_t* x1_indices,
                const T* x2_data, const index_t* x2_indices,
                index_t x1_start, index_t x1_end,
                index_t x2_start, index_t x2_end,
                size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double d = 0.0, unsquared = 0.0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) {
                unsquared = x1_data[i1] - x2_data[i2];
                d += unsquared * unsquared / vec[ix1];
                ++i1; ++i2;
            } else if (ix1 < ix2) {
                unsquared = x1_data[i1]; d += unsquared * unsquared / vec[ix1]; ++i1;
            } else {
                unsquared = x2_data[i2]; d += unsquared * unsquared / vec[ix2]; ++i2;
            }
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) {
                index_t ix2 = x2_indices[i2];
                unsquared = x2_data[i2]; d += unsquared * unsquared / vec[ix2]; ++i2;
            }
        } else {
            while (i1 < x1_end) {
                index_t ix1 = x1_indices[i1];
                unsquared = x1_data[i1]; d += unsquared * unsquared / vec[ix1]; ++i1;
            }
        }
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        return static_cast<T>(sqrt(static_cast<double>(rdist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, size))));
    }
};

// ---------------------------------------------------------------------------
// Manhattan / cityblock:  d = sum |x_i - y_i|
// (dense accumulates in double; CSR accumulates in T -- faithful to Cython)
// ---------------------------------------------------------------------------
template <typename T>
struct ManhattanDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        double d = 0.0;
        for (size_type j = 0; j < size; ++j) {
            double diff = x1[j] - x2[j];
            d += fabs(diff);
        }
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        T d = 0;  // FAITHFUL: Cython accumulates Manhattan CSR in {{INPUT_DTYPE_t}}
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) {
                d += static_cast<T>(fabs(static_cast<double>(x1_data[i1] - x2_data[i2]))); ++i1; ++i2;
            } else if (ix1 < ix2) {
                d += static_cast<T>(fabs(static_cast<double>(x1_data[i1]))); ++i1;
            } else {
                d += static_cast<T>(fabs(static_cast<double>(x2_data[i2]))); ++i2;
            }
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) { d += static_cast<T>(fabs(static_cast<double>(x2_data[i2]))); ++i2; }
        } else {
            while (i1 < x1_end) { d += static_cast<T>(fabs(static_cast<double>(x1_data[i1]))); ++i1; }
        }
        return d;
    }
};

// ---------------------------------------------------------------------------
// Chebyshev / infinity:  d = max |x_i - y_i|
// ---------------------------------------------------------------------------
template <typename T>
struct ChebyshevDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        double d = 0.0;
        for (size_type j = 0; j < size; ++j)
            d = std::max(d, fabs(static_cast<double>(x1[j] - x2[j])));
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double d = 0.0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) {
                d = std::max(d, fabs(static_cast<double>(x1_data[i1] - x2_data[i2]))); ++i1; ++i2;
            } else if (ix1 < ix2) {
                d = std::max(d, fabs(static_cast<double>(x1_data[i1]))); ++i1;
            } else {
                d = std::max(d, fabs(static_cast<double>(x2_data[i2]))); ++i2;
            }
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) { d = std::max(d, fabs(static_cast<double>(x2_data[i2]))); ++i2; }
        } else {
            while (i1 < x1_end) { d = std::max(d, fabs(static_cast<double>(x1_data[i1]))); ++i1; }
        }
        return static_cast<T>(d);
    }
};

// ---------------------------------------------------------------------------
// Minkowski:  rdist = sum [w_i] |x_i - y_i|^p ;  d = rdist^(1/p)
// ---------------------------------------------------------------------------
template <typename T>
struct MinkowskiDistance : MetricBase<T> {
    double p;
    std::vector<double> w;  // empty -> unweighted
    MinkowskiDistance(double p_, const double* w_ptr, size_type w_len)
        : p(p_), w(w_ptr ? std::vector<double>(w_ptr, w_ptr + w_len) : std::vector<double>()) {}
    bool has_w() const { return !w.empty(); }

    T rdist(const T* x1, const T* x2, size_type size) const override {
        double d = 0.0;
        if (has_w()) {
            for (size_type j = 0; j < size; ++j)
                d += w[j] * pow(fabs(static_cast<double>(x1[j] - x2[j])), p);
        } else {
            for (size_type j = 0; j < size; ++j)
                d += pow(fabs(static_cast<double>(x1[j] - x2[j])), p);
        }
        return static_cast<T>(d);
    }
    T dist(const T* x1, const T* x2, size_type size) const override {
        return static_cast<T>(pow(static_cast<double>(rdist(x1, x2, size)), 1.0 / p));
    }
    T rdist_to_dist(T rdist) const override { return static_cast<T>(pow(static_cast<double>(rdist), 1.0 / p)); }
    T dist_to_rdist(T dist) const override { return static_cast<T>(pow(static_cast<double>(dist), p)); }

    T rdist_csr(const T* x1_data, const index_t* x1_indices,
                const T* x2_data, const index_t* x2_indices,
                index_t x1_start, index_t x1_end,
                index_t x2_start, index_t x2_end,
                size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double d = 0.0;
        if (has_w()) {
            while (i1 < x1_end && i2 < x2_end) {
                index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
                if (ix1 == ix2) {
                    d += w[ix1] * pow(fabs(static_cast<double>(x1_data[i1] - x2_data[i2])), p); ++i1; ++i2;
                } else if (ix1 < ix2) {
                    d += w[ix1] * pow(fabs(static_cast<double>(x1_data[i1])), p); ++i1;
                } else {
                    d += w[ix2] * pow(fabs(static_cast<double>(x2_data[i2])), p); ++i2;
                }
            }
            if (i1 == x1_end) {
                while (i2 < x2_end) { index_t ix2 = x2_indices[i2]; d += w[ix2] * pow(fabs(static_cast<double>(x2_data[i2])), p); ++i2; }
            } else {
                while (i1 < x1_end) { index_t ix1 = x1_indices[i1]; d += w[ix1] * pow(fabs(static_cast<double>(x1_data[i1])), p); ++i1; }
            }
        } else {
            while (i1 < x1_end && i2 < x2_end) {
                index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
                if (ix1 == ix2) {
                    d += pow(fabs(static_cast<double>(x1_data[i1] - x2_data[i2])), p); ++i1; ++i2;
                } else if (ix1 < ix2) {
                    d += pow(fabs(static_cast<double>(x1_data[i1])), p); ++i1;
                } else {
                    d += pow(fabs(static_cast<double>(x2_data[i2])), p); ++i2;
                }
            }
            if (i1 == x1_end) {
                while (i2 < x2_end) { d += pow(fabs(static_cast<double>(x2_data[i2])), p); ++i2; }
            } else {
                while (i1 < x1_end) { d += pow(fabs(static_cast<double>(x1_data[i1])), p); ++i1; }
            }
        }
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        return static_cast<T>(pow(static_cast<double>(rdist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, size)), 1.0 / p));
    }
};

// ---------------------------------------------------------------------------
// Mahalanobis:  d = sqrt( (x - y)^T VI (x - y) )
// NOTE: `buffer` is a per-instance scratch vector and is NOT reentrant. This
// matches the Cython implementation, which is only ever used single-threaded
// (Mahalanobis is excluded from the parallel pairwise-distances reductions).
// ---------------------------------------------------------------------------
template <typename T>
struct MahalanobisDistance : MetricBase<T> {
    std::vector<double> mat;  // VI, row-major size*size
    size_type size;
    mutable std::vector<double> buffer;
    MahalanobisDistance(const double* VI, size_type n)
        : mat(VI, VI + n * n), size(n), buffer(n, 0.0) {}

    T rdist(const T* x1, const T* x2, size_type sz) const override {
        for (size_type i = 0; i < sz; ++i) buffer[i] = x1[i] - x2[i];
        double d = 0.0;
        for (size_type i = 0; i < sz; ++i) {
            double tmp = 0.0;
            for (size_type j = 0; j < sz; ++j) tmp += mat[i * size + j] * buffer[j];
            d += tmp * buffer[i];
        }
        return static_cast<T>(d);
    }
    T dist(const T* x1, const T* x2, size_type sz) const override {
        return static_cast<T>(sqrt(static_cast<double>(rdist(x1, x2, sz))));
    }
    T rdist_to_dist(T rdist) const override { return static_cast<T>(sqrt(static_cast<double>(rdist))); }
    T dist_to_rdist(T dist) const override { return static_cast<T>(static_cast<double>(dist) * dist); }

    T rdist_csr(const T* x1_data, const index_t* x1_indices,
                const T* x2_data, const index_t* x2_indices,
                index_t x1_start, index_t x1_end,
                index_t x2_start, index_t x2_end,
                size_type sz) const override {
        // FAITHFUL: the Cython version does not zero `buffer` here; entries for
        // dimensions absent from both x1 and x2 retain their previous value.
        index_t i1 = x1_start, i2 = x2_start;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) { buffer[ix1] = x1_data[i1] - x2_data[i2]; ++i1; ++i2; }
            else if (ix1 < ix2) { buffer[ix1] = x1_data[i1]; ++i1; }
            else { buffer[ix2] = -static_cast<double>(x2_data[i2]); ++i2; }
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) { index_t ix2 = x2_indices[i2]; buffer[ix2] = -static_cast<double>(x2_data[i2]); ++i2; }
        } else {
            while (i1 < x1_end) { index_t ix1 = x1_indices[i1]; buffer[ix1] = x1_data[i1]; ++i1; }
        }
        double d = 0.0;
        for (size_type i = 0; i < sz; ++i) {
            double tmp = 0.0;
            for (size_type j = 0; j < sz; ++j) tmp += mat[i * size + j] * buffer[j];
            d += tmp * buffer[i];
        }
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type sz) const override {
        return static_cast<T>(sqrt(static_cast<double>(rdist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, sz))));
    }
};

// ---------------------------------------------------------------------------
// Hamming:  d = N_unequal / N
// ---------------------------------------------------------------------------
template <typename T>
struct HammingDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        int n_unequal = 0;
        for (size_type j = 0; j < size; ++j)
            if (x1[j] != x2[j]) ++n_unequal;
        return static_cast<T>(static_cast<double>(n_unequal) / size);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double d = 0.0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) { d += (x1_data[i1] != x2_data[i2]); ++i1; ++i2; }
            else if (ix1 < ix2) { d += (x1_data[i1] != 0); ++i1; }
            else { d += (x2_data[i2] != 0); ++i2; }
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) { d += (x2_data[i2] != 0); ++i2; }
        } else {
            while (i1 < x1_end) { d += (x1_data[i1] != 0); ++i1; }
        }
        d /= size;
        return static_cast<T>(d);
    }
};

// ---------------------------------------------------------------------------
// Canberra:  d = sum |x_i - y_i| / (|x_i| + |y_i|)
// ---------------------------------------------------------------------------
template <typename T>
struct CanberraDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        double d = 0.0;
        for (size_type j = 0; j < size; ++j) {
            double denom = fabs(static_cast<double>(x1[j])) + fabs(static_cast<double>(x2[j]));
            if (denom > 0) d += fabs(static_cast<double>(x1[j] - x2[j])) / denom;
        }
        return static_cast<T>(d);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double d = 0.0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) {
                d += fabs(static_cast<double>(x1_data[i1] - x2_data[i2])) /
                     (fabs(static_cast<double>(x1_data[i1])) + fabs(static_cast<double>(x2_data[i2])));
                ++i1; ++i2;
            } else if (ix1 < ix2) { d += 1.0; ++i1; }
            else { d += 1.0; ++i2; }
        }
        if (i1 == x1_end) { while (i2 < x2_end) { d += 1.0; ++i2; } }
        else { while (i1 < x1_end) { d += 1.0; ++i1; } }
        return static_cast<T>(d);
    }
};

// ---------------------------------------------------------------------------
// Bray-Curtis:  d = sum|x_i - y_i| / sum(|x_i| + |y_i|)
// ---------------------------------------------------------------------------
template <typename T>
struct BrayCurtisDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        double num = 0.0, denom = 0.0;
        for (size_type j = 0; j < size; ++j) {
            num += fabs(static_cast<double>(x1[j] - x2[j]));
            denom += fabs(static_cast<double>(x1[j])) + fabs(static_cast<double>(x2[j]));
        }
        return static_cast<T>(denom > 0 ? num / denom : 0.0);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double num = 0.0, denom = 0.0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            if (ix1 == ix2) {
                num += fabs(static_cast<double>(x1_data[i1] - x2_data[i2]));
                denom += fabs(static_cast<double>(x1_data[i1])) + fabs(static_cast<double>(x2_data[i2]));
                ++i1; ++i2;
            } else if (ix1 < ix2) {
                num += fabs(static_cast<double>(x1_data[i1]));
                denom += fabs(static_cast<double>(x1_data[i1]));
                ++i1;
            } else {
                num += fabs(static_cast<double>(x2_data[i2]));
                denom += fabs(static_cast<double>(x2_data[i2]));
                ++i2;
            }
        }
        // FAITHFUL: the Cython tail loops read the opposite vector's data and
        // advance the opposite index; reproduced exactly to preserve behavior.
        if (i1 == x1_end) {
            while (i2 < x2_end) {
                num += fabs(static_cast<double>(x1_data[i1]));
                denom += fabs(static_cast<double>(x1_data[i1]));
                ++i2;
            }
        } else {
            while (i1 < x1_end) {
                num += fabs(static_cast<double>(x2_data[i2]));
                denom += fabs(static_cast<double>(x2_data[i2]));
                ++i1;
            }
        }
        return static_cast<T>(num / denom);
    }
};

// ---------------------------------------------------------------------------
// Boolean metrics. Any nonzero entry is treated as True.
// Helper to count, over the merged CSR supports, the True/True (n_tt),
// not-equal (n_neq) and nonzero-union (nnz) totals.
// ---------------------------------------------------------------------------
namespace detail {
struct BoolCounts { long n_tt = 0; long n_neq = 0; long nnz = 0; };

template <typename T>
inline BoolCounts csr_bool_counts(const T* x1_data, const index_t* x1_indices,
                                  const T* x2_data, const index_t* x2_indices,
                                  index_t x1_start, index_t x1_end,
                                  index_t x2_start, index_t x2_end) {
    BoolCounts c;
    index_t i1 = x1_start, i2 = x2_start;
    while (i1 < x1_end && i2 < x2_end) {
        index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
        long tf1 = x1_data[i1] != 0, tf2 = x2_data[i2] != 0;
        if (ix1 == ix2) {
            c.nnz += (tf1 || tf2); c.n_tt += (tf1 && tf2); c.n_neq += (tf1 != tf2);
            ++i1; ++i2;
        } else if (ix1 < ix2) {
            c.nnz += tf1; c.n_neq += tf1; ++i1;
        } else {
            c.nnz += tf2; c.n_neq += tf2; ++i2;
        }
    }
    if (i1 == x1_end) {
        while (i2 < x2_end) { long tf2 = x2_data[i2] != 0; c.nnz += tf2; c.n_neq += tf2; ++i2; }
    } else {
        while (i1 < x1_end) { long tf1 = x1_data[i1] != 0; c.nnz += tf1; c.n_neq += tf1; ++i1; }
    }
    return c;
}

template <typename T>
inline void dense_bool_counts(const T* x1, const T* x2, size_type size,
                              long& n_tt, long& n_neq, long& nnz) {
    n_tt = n_neq = nnz = 0;
    for (size_type j = 0; j < size; ++j) {
        long tf1 = x1[j] != 0, tf2 = x2[j] != 0;
        n_tt += (tf1 && tf2); n_neq += (tf1 != tf2); nnz += (tf1 || tf2);
    }
}
}  // namespace detail

// Jaccard:  (nnz - n_tt) / nnz   (0 when nnz == 0)
template <typename T>
struct JaccardDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_eq = 0, nnz = 0;
        for (size_type j = 0; j < size; ++j) {
            long tf1 = x1[j] != 0, tf2 = x2[j] != 0;
            nnz += (tf1 || tf2); n_eq += (tf1 && tf2);
        }
        if (nnz == 0) return 0;
        return static_cast<T>((nnz - n_eq) * 1.0 / nnz);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        if (c.nnz == 0) return 0;
        return static_cast<T>((c.nnz - c.n_tt) * 1.0 / c.nnz);
    }
};

// Matching:  n_neq / N
template <typename T>
struct MatchingDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>(n_neq * 1.0 / size);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        return static_cast<T>(c.n_neq * 1.0 / size);
    }
};

// Dice:  n_neq / (2 n_tt + n_neq)
template <typename T>
struct DiceDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>(n_neq / (2.0 * n_tt + n_neq));
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        return static_cast<T>(c.n_neq / (2.0 * c.n_tt + c.n_neq));
    }
};

// Kulsinski:  (n_neq - n_tt + N) / (n_neq + N)
template <typename T>
struct KulsinskiDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>((n_neq - n_tt + size) * 1.0 / (n_neq + size));
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        return static_cast<T>((c.n_neq - c.n_tt + size) * 1.0 / (c.n_neq + size));
    }
};

// Rogers-Tanimoto:  2 n_neq / (N + n_neq)
template <typename T>
struct RogersTanimotoDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>((2.0 * n_neq) / (size + n_neq));
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        return static_cast<T>((2.0 * c.n_neq) / (size + c.n_neq));
    }
};

// Russell-Rao:  (N - n_tt) / N
template <typename T>
struct RussellRaoDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>((size - n_tt) * 1.0 / size);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        // FAITHFUL: only the merged prefix is walked (matches Cython: once one
        // vector is exhausted, n_tt can no longer increase, so the tail is skipped).
        index_t i1 = x1_start, i2 = x2_start;
        long n_tt = 0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            long tf1 = x1_data[i1] != 0, tf2 = x2_data[i2] != 0;
            if (ix1 == ix2) { n_tt += (tf1 && tf2); ++i1; ++i2; }
            else if (ix1 < ix2) { ++i1; }
            else { ++i2; }
        }
        return static_cast<T>((size - n_tt) * 1.0 / size);
    }
};

// Sokal-Michener:  2 n_neq / (N + n_neq)
template <typename T>
struct SokalMichenerDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>((2.0 * n_neq) / (size + n_neq));
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        return static_cast<T>((2.0 * c.n_neq) / (size + c.n_neq));
    }
};

// Sokal-Sneath:  n_neq / (0.5 n_tt + n_neq)
template <typename T>
struct SokalSneathDistance : MetricBase<T> {
    T dist(const T* x1, const T* x2, size_type size) const override {
        long n_tt, n_neq, nnz; detail::dense_bool_counts(x1, x2, size, n_tt, n_neq, nnz);
        return static_cast<T>(n_neq / (0.5 * n_tt + n_neq));
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type /*size*/) const override {
        auto c = detail::csr_bool_counts(x1_data, x1_indices, x2_data, x2_indices,
                                         x1_start, x1_end, x2_start, x2_end);
        return static_cast<T>(c.n_neq / (0.5 * c.n_tt + c.n_neq));
    }
};

// ---------------------------------------------------------------------------
// Haversine (2D lat/long in radians)
// ---------------------------------------------------------------------------
template <typename T>
struct HaversineDistance : MetricBase<T> {
    T rdist(const T* x1, const T* x2, size_type /*size*/) const override {
        double sin_0 = sin(0.5 * (static_cast<double>(x1[0]) - x2[0]));
        double sin_1 = sin(0.5 * (static_cast<double>(x1[1]) - x2[1]));
        return static_cast<T>(sin_0 * sin_0 + cos(static_cast<double>(x1[0])) * cos(static_cast<double>(x2[0])) * sin_1 * sin_1);
    }
    T dist(const T* x1, const T* x2, size_type size) const override {
        return static_cast<T>(2 * asin(sqrt(static_cast<double>(rdist(x1, x2, size)))));
    }
    T rdist_to_dist(T rdist) const override { return static_cast<T>(2 * asin(sqrt(static_cast<double>(rdist)))); }
    T dist_to_rdist(T dist) const override { double tmp = sin(0.5 * static_cast<double>(dist)); return static_cast<T>(tmp * tmp); }

    T rdist_csr(const T* x1_data, const index_t* x1_indices,
                const T* x2_data, const index_t* x2_indices,
                index_t x1_start, index_t x1_end,
                index_t x2_start, index_t x2_end,
                size_type /*size*/) const override {
        index_t i1 = x1_start, i2 = x2_start;
        double x1_0 = 0, x1_1 = 0, x2_0 = 0, x2_1 = 0;
        while (i1 < x1_end && i2 < x2_end) {
            index_t ix1 = x1_indices[i1], ix2 = x2_indices[i2];
            index_t x1_component = (x1_start == 0) ? ix1 : ix1 % x1_start;
            index_t x2_component = (x2_start == 0) ? ix2 : ix2 % x2_start;
            if (x1_component == 0) x1_0 = x1_data[i1]; else x1_1 = x1_data[i1];
            if (x2_component == 0) x2_0 = x2_data[i2]; else x2_1 = x2_data[i2];
            ++i1; ++i2;
        }
        if (i1 == x1_end) {
            while (i2 < x2_end) {
                index_t ix2 = x2_indices[i2];
                index_t x2_component = (x2_start == 0) ? ix2 : ix2 % x2_start;
                if (x2_component == 0) x2_0 = x2_data[i2]; else x2_1 = x2_data[i2];
                ++i2;
            }
        } else {
            while (i1 < x1_end) {
                index_t ix1 = x1_indices[i1];
                index_t x1_component = (x1_start == 0) ? ix1 : ix1 % x1_start;
                if (x1_component == 0) x1_0 = x1_data[i1]; else x1_1 = x1_data[i1];
                ++i1;
            }
        }
        double sin_0 = sin(0.5 * (x1_0 - x2_0));
        double sin_1 = sin(0.5 * (x1_1 - x2_1));
        return static_cast<T>(sin_0 * sin_0 + cos(x1_0) * cos(x2_0) * sin_1 * sin_1);
    }
    T dist_csr(const T* x1_data, const index_t* x1_indices,
               const T* x2_data, const index_t* x2_indices,
               index_t x1_start, index_t x1_end,
               index_t x2_start, index_t x2_end,
               size_type size) const override {
        return static_cast<T>(2 * asin(sqrt(static_cast<double>(rdist_csr(
            x1_data, x1_indices, x2_data, x2_indices,
            x1_start, x1_end, x2_start, x2_end, size)))));
    }
};

// ---------------------------------------------------------------------------
// Factory helpers (return owning raw pointers; caller deletes).
// Templated on T so Cython can call e.g. make_euclidean[float64_t]().
// ---------------------------------------------------------------------------
template <typename T> MetricBase<T>* make_euclidean() { return new EuclideanDistance<T>(); }
template <typename T> MetricBase<T>* make_seuclidean(const double* V, size_type n) { return new SEuclideanDistance<T>(V, n); }
template <typename T> MetricBase<T>* make_manhattan() { return new ManhattanDistance<T>(); }
template <typename T> MetricBase<T>* make_chebyshev() { return new ChebyshevDistance<T>(); }
template <typename T> MetricBase<T>* make_minkowski(double p, const double* w, size_type w_len) { return new MinkowskiDistance<T>(p, w, w_len); }
template <typename T> MetricBase<T>* make_mahalanobis(const double* VI, size_type n) { return new MahalanobisDistance<T>(VI, n); }
template <typename T> MetricBase<T>* make_hamming() { return new HammingDistance<T>(); }
template <typename T> MetricBase<T>* make_canberra() { return new CanberraDistance<T>(); }
template <typename T> MetricBase<T>* make_braycurtis() { return new BrayCurtisDistance<T>(); }
template <typename T> MetricBase<T>* make_jaccard() { return new JaccardDistance<T>(); }
template <typename T> MetricBase<T>* make_matching() { return new MatchingDistance<T>(); }
template <typename T> MetricBase<T>* make_dice() { return new DiceDistance<T>(); }
template <typename T> MetricBase<T>* make_kulsinski() { return new KulsinskiDistance<T>(); }
template <typename T> MetricBase<T>* make_rogerstanimoto() { return new RogersTanimotoDistance<T>(); }
template <typename T> MetricBase<T>* make_russellrao() { return new RussellRaoDistance<T>(); }
template <typename T> MetricBase<T>* make_sokalmichener() { return new SokalMichenerDistance<T>(); }
template <typename T> MetricBase<T>* make_sokalsneath() { return new SokalSneathDistance<T>(); }
template <typename T> MetricBase<T>* make_haversine() { return new HaversineDistance<T>(); }

}  // namespace metrics
}  // namespace sklearn

#endif  // SKLEARN_METRICS_METRIC_KERNELS_HPP
