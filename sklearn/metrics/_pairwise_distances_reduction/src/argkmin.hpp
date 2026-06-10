/* ArgKmin: for each row of X, the indices (and optionally distances) of the k
 * nearest rows of Y. C++ counterpart of ArgKmin{32,64} and EuclideanArgKmin{32,64}
 * in _argkmin.pyx.tp.
 *
 * ArgKminHeaps<T> holds the shared machinery (per-row fixed-size max-heaps, the
 * two parallel strategies' heap hooks, result buffers, exact-distance
 * conversion). Two reductions build on it:
 *   - ArgKmin<T>: generic, evaluates each pair's surrogate distance through a
 *     DatasetsPair<T> (dense or any CSR combination).
 *   - EuclideanArgKmin<T>: the GEMM specialization -- the squared distance of a
 *     chunk is ||x||^2 - 2 x.y^T + ||y||^2, the middle term computed with one
 *     dgemm per chunk (float32 inputs upcast to double first).
 */
#ifndef SKLEARN_PDR_ARGKMIN_HPP
#define SKLEARN_PDR_ARGKMIN_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "metric_kernels.hpp"
#include "base.hpp"
#include "datasets_pair.hpp"
#include "heap.hpp"
#include "middle_term.hpp"

namespace sklearn {
namespace pdr {

using sklearn::metrics::MetricBase;

template <typename T>
struct ArgKminHeaps : ReductionHooks {
    idx_t n_samples_X;
    idx_t n_samples_Y;
    idx_t k;
    bool use_squared_distances;
    const MetricBase<T>* metric;  // borrowed; used for rdist_to_dist
    const ChunkingConfig& cfg;

    // Results, row-major (n_samples_X, k). Heaps hold double regardless of T.
    std::vector<double> argkmin_distances;
    std::vector<idx_t> argkmin_indices;

    // Per-thread heap pointers. parallel_on_X: into the main result arrays;
    // parallel_on_Y: into the per-thread scratch buffers below.
    std::vector<double*> heaps_r;
    std::vector<idx_t*> heaps_i;
    std::vector<std::vector<double>> heaps_r_buf;
    std::vector<std::vector<idx_t>> heaps_i_buf;

    ArgKminHeaps(idx_t n_samples_X_, idx_t n_samples_Y_, idx_t k_,
                 bool use_squared_distances_, const MetricBase<T>* metric_,
                 const ChunkingConfig& cfg_)
        : n_samples_X(n_samples_X_), n_samples_Y(n_samples_Y_), k(k_),
          use_squared_distances(use_squared_distances_), metric(metric_), cfg(cfg_) {
        argkmin_distances.assign(n_samples_X * k, std::numeric_limits<double>::max());
        argkmin_indices.assign(n_samples_X * k, 0);
        heaps_r.resize(cfg.chunks_n_threads);
        heaps_i.resize(cfg.chunks_n_threads);
    }

    // ---- parallel_on_X heap hooks ----
    void parallel_on_X_init_chunk(int tn, idx_t X_start, idx_t /*X_end*/) {
        heaps_r[tn] = &argkmin_distances[X_start * k];
        heaps_i[tn] = &argkmin_indices[X_start * k];
    }
    void parallel_on_X_prange_iter_finalize(int tn, idx_t X_start, idx_t X_end) {
        for (idx_t idx = 0; idx < X_end - X_start; ++idx)
            simultaneous_sort(heaps_r[tn] + idx * k, heaps_i[tn] + idx * k, k);
    }

    // ---- parallel_on_Y heap hooks ----
    void parallel_on_Y_init() {
        idx_t heaps_size = cfg.X_n_samples_chunk * k;
        heaps_r_buf.assign(cfg.chunks_n_threads, std::vector<double>(heaps_size));
        heaps_i_buf.assign(cfg.chunks_n_threads, std::vector<idx_t>(heaps_size));
        for (idx_t tn = 0; tn < cfg.chunks_n_threads; ++tn) {
            heaps_r[tn] = heaps_r_buf[tn].data();
            heaps_i[tn] = heaps_i_buf[tn].data();
        }
    }
    void parallel_on_Y_parallel_init(int tn, idx_t /*X_start*/, idx_t /*X_end*/) {
        idx_t sz = cfg.X_n_samples_chunk * k;
        for (idx_t idx = 0; idx < sz; ++idx) {
            heaps_r[tn][idx] = std::numeric_limits<double>::max();
            heaps_i[tn][idx] = -1;
        }
    }
    void parallel_on_Y_synchronize(idx_t X_start, idx_t X_end) {
        #pragma omp parallel for schedule(static) num_threads(cfg.effective_n_threads)
        for (idx_t idx = 0; idx < X_end - X_start; ++idx) {
            for (idx_t tn = 0; tn < cfg.chunks_n_threads; ++tn)
                for (idx_t jdx = 0; jdx < k; ++jdx)
                    heap_push(&argkmin_distances[(X_start + idx) * k],
                              &argkmin_indices[(X_start + idx) * k], k,
                              heaps_r[tn][idx * k + jdx], heaps_i[tn][idx * k + jdx]);
        }
    }
    void parallel_on_Y_finalize() {
        #pragma omp parallel for schedule(static) num_threads(cfg.effective_n_threads)
        for (idx_t idx = 0; idx < n_samples_X; ++idx)
            simultaneous_sort(&argkmin_distances[idx * k], &argkmin_indices[idx * k], k);
    }

    // Convert the rank-preserving (squared) distances to exact distances.
    void compute_exact_distances() {
        if (use_squared_distances) return;
        #pragma omp parallel for schedule(static) num_threads(cfg.effective_n_threads)
        for (idx_t i = 0; i < n_samples_X; ++i)
            for (idx_t j = 0; j < k; ++j) {
                // Guard against -0. (catastrophic cancellation) producing NaN.
                double r = std::max(argkmin_distances[i * k + j], 0.0);
                argkmin_distances[i * k + j] = metric->rdist_to_dist(static_cast<T>(r));
            }
    }
};

// ---------------------------------------------------------------------------
// Generic per-pair ArgKmin
// ---------------------------------------------------------------------------
template <typename T>
struct ArgKmin : ArgKminHeaps<T> {
    DatasetsPair<T> dp;

    ArgKmin(DatasetsPair<T> dp_, idx_t n_samples_X_, idx_t n_samples_Y_, idx_t k_,
            bool use_squared_distances_, const ChunkingConfig& cfg_)
        : ArgKminHeaps<T>(n_samples_X_, n_samples_Y_, k_, use_squared_distances_,
                          dp_.metric, cfg_),
          dp(dp_) {}

    void compute_and_reduce(idx_t X_start, idx_t X_end, idx_t Y_start, idx_t Y_end, int tn) {
        idx_t nX = X_end - X_start, nY = Y_end - Y_start, k = this->k;
        double* hr = this->heaps_r[tn];
        idx_t* hi = this->heaps_i[tn];
        for (idx_t i = 0; i < nX; ++i)
            for (idx_t j = 0; j < nY; ++j)
                heap_push(hr + i * k, hi + i * k, k,
                          dp.rdist(X_start + i, Y_start + j), Y_start + j);
    }
};

// ---------------------------------------------------------------------------
// Euclidean GEMM specialization (dense-dense)
// ---------------------------------------------------------------------------
template <typename T>
struct EuclideanArgKmin : ArgKminHeaps<T> {
    const T* X;
    const T* Y;
    idx_t n_features;
    std::vector<double> X_norm;   // ||X_i||^2
    std::vector<double> Y_norm;   // ||Y_j||^2
    dgemm_t dgemm;

    // Per-thread scratch: chunk middle term, and (float32 only) upcast chunks.
    std::vector<std::vector<double>> middle;
    std::vector<std::vector<double>> Xup;
    std::vector<std::vector<double>> Yup;

    static constexpr bool needs_upcast = !std::is_same<T, double>::value;

    EuclideanArgKmin(const T* X_, const T* Y_, idx_t n_features_,
                     idx_t n_samples_X_, idx_t n_samples_Y_, idx_t k_,
                     bool use_squared_distances_, const MetricBase<T>* metric_,
                     std::vector<double> X_norm_, std::vector<double> Y_norm_,
                     dgemm_t dgemm_, const ChunkingConfig& cfg_)
        : ArgKminHeaps<T>(n_samples_X_, n_samples_Y_, k_, use_squared_distances_,
                          metric_, cfg_),
          X(X_), Y(Y_), n_features(n_features_),
          X_norm(std::move(X_norm_)), Y_norm(std::move(Y_norm_)), dgemm(dgemm_) {
        idx_t nt = cfg_.chunks_n_threads;
        middle.assign(nt, std::vector<double>(cfg_.X_n_samples_chunk * cfg_.Y_n_samples_chunk));
        if (needs_upcast) {
            Xup.assign(nt, std::vector<double>(cfg_.X_n_samples_chunk * n_features));
            Yup.assign(nt, std::vector<double>(cfg_.Y_n_samples_chunk * n_features));
        }
    }

    inline void upcast_chunk(const T* src, idx_t start, idx_t n_rows, std::vector<double>& dst) {
        const T* p = src + start * n_features;
        idx_t total = n_rows * n_features;
        for (idx_t i = 0; i < total; ++i) dst[i] = static_cast<double>(p[i]);
    }

    // ---- parallel_on_X: heap init + X upcast / Y upcast ----
    void parallel_on_X_init_chunk(int tn, idx_t X_start, idx_t X_end) {
        ArgKminHeaps<T>::parallel_on_X_init_chunk(tn, X_start, X_end);
        if constexpr (needs_upcast)
            upcast_chunk(X, X_start, X_end - X_start, Xup[tn]);
    }
    void parallel_on_X_pre_compute(idx_t, idx_t, idx_t Y_start, idx_t Y_end, int tn) {
        if constexpr (needs_upcast)
            upcast_chunk(Y, Y_start, Y_end - Y_start, Yup[tn]);
    }

    // ---- parallel_on_Y: heap reset + X upcast / Y upcast ----
    void parallel_on_Y_parallel_init(int tn, idx_t X_start, idx_t X_end) {
        ArgKminHeaps<T>::parallel_on_Y_parallel_init(tn, X_start, X_end);
        if constexpr (needs_upcast)
            upcast_chunk(X, X_start, X_end - X_start, Xup[tn]);
    }
    void parallel_on_Y_pre_compute(idx_t, idx_t, idx_t Y_start, idx_t Y_end, int tn) {
        if constexpr (needs_upcast)
            upcast_chunk(Y, Y_start, Y_end - Y_start, Yup[tn]);
    }

    void compute_and_reduce(idx_t X_start, idx_t X_end, idx_t Y_start, idx_t Y_end, int tn) {
        idx_t nX = X_end - X_start, nY = Y_end - Y_start, k = this->k;

        const double* A;
        const double* B;
        if constexpr (needs_upcast) {
            A = Xup[tn].data();
            B = Yup[tn].data();
        } else {
            A = X + X_start * n_features;
            B = Y + Y_start * n_features;
        }
        double* M = middle[tn].data();
        gemm_minus2_XYt(dgemm, A, B, M, static_cast<int>(nX), static_cast<int>(nY),
                        static_cast<int>(n_features));

        double* hr = this->heaps_r[tn];
        idx_t* hi = this->heaps_i[tn];
        for (idx_t i = 0; i < nX; ++i) {
            double xn = X_norm[X_start + i];
            for (idx_t j = 0; j < nY; ++j) {
                double sq = xn + M[i * nY + j] + Y_norm[Y_start + j];
                // Catastrophic cancellation can yield small negatives (e.g. X is Y).
                if (sq < 0.0) sq = 0.0;
                heap_push(hr + i * k, hi + i * k, k, sq, Y_start + j);
            }
        }
    }
};

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_ARGKMIN_HPP
