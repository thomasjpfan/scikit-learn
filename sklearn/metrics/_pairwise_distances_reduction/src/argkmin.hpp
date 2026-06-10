/* ArgKmin: for each row of X, the indices (and optionally distances) of the k
 * nearest rows of Y. C++ counterpart of ArgKmin{32,64} in _argkmin.pyx.tp.
 *
 * This is the generic, per-pair implementation: the surrogate (rank-preserving)
 * distance of each (i, j) pair is obtained from a MetricBase<T> functor and
 * pushed onto a per-row fixed-size max-heap. The Euclidean GEMM specialization
 * is added in a later step; here even the Euclidean metric goes through the
 * functor (its rdist is the squared Euclidean distance), which keeps the result
 * identical up to floating-point reassociation.
 */
#ifndef SKLEARN_PDR_ARGKMIN_HPP
#define SKLEARN_PDR_ARGKMIN_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "metric_kernels.hpp"
#include "base.hpp"
#include "datasets_pair.hpp"
#include "heap.hpp"

namespace sklearn {
namespace pdr {

using sklearn::metrics::MetricBase;

template <typename T>
struct ArgKmin : ReductionHooks {
    DatasetsPair<T> dp;            // distance source (dense or any CSR combination)
    idx_t n_samples_X;
    idx_t n_samples_Y;
    idx_t k;
    bool use_squared_distances;
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

    ArgKmin(DatasetsPair<T> dp_, idx_t n_samples_X_, idx_t n_samples_Y_, idx_t k_,
            bool use_squared_distances_, const ChunkingConfig& cfg_)
        : dp(dp_), n_samples_X(n_samples_X_), n_samples_Y(n_samples_Y_), k(k_),
          use_squared_distances(use_squared_distances_), cfg(cfg_) {
        argkmin_distances.assign(n_samples_X * k, std::numeric_limits<double>::max());
        argkmin_indices.assign(n_samples_X * k, 0);
        heaps_r.resize(cfg.chunks_n_threads);
        heaps_i.resize(cfg.chunks_n_threads);
    }

    inline double surrogate(idx_t i, idx_t j) const { return dp.rdist(i, j); }

    void compute_and_reduce(idx_t X_start, idx_t X_end, idx_t Y_start, idx_t Y_end, int tn) {
        idx_t nX = X_end - X_start, nY = Y_end - Y_start;
        double* hr = heaps_r[tn];
        idx_t* hi = heaps_i[tn];
        for (idx_t i = 0; i < nX; ++i)
            for (idx_t j = 0; j < nY; ++j)
                heap_push(hr + i * k, hi + i * k, k,
                          surrogate(X_start + i, Y_start + j), Y_start + j);
    }

    // ---- parallel_on_X ----
    void parallel_on_X_init_chunk(int tn, idx_t X_start, idx_t /*X_end*/) {
        heaps_r[tn] = &argkmin_distances[X_start * k];
        heaps_i[tn] = &argkmin_indices[X_start * k];
    }
    void parallel_on_X_prange_iter_finalize(int tn, idx_t X_start, idx_t X_end) {
        for (idx_t idx = 0; idx < X_end - X_start; ++idx)
            simultaneous_sort(heaps_r[tn] + idx * k, heaps_i[tn] + idx * k, k);
    }

    // ---- parallel_on_Y ----
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
                argkmin_distances[i * k + j] = dp.metric->rdist_to_dist(static_cast<T>(r));
            }
    }
};

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_ARGKMIN_HPP
