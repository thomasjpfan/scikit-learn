/* RadiusNeighbors: for each row of X, the indices (and optionally distances) of
 * the rows of Y within `radius`. C++ counterpart of RadiusNeighbors{32,64} in
 * _radius_neighbors.pyx.tp.
 *
 * Results are ragged (variable count per row). They are accumulated as per-row
 * vectors and flattened to (indices_flat, distances_flat, indptr) by the binding;
 * the Python dispatcher then builds the object-array of per-row views.
 *
 * Generic per-pair implementation: the surrogate distance comes from a
 * DatasetsPair<T> (dense or any CSR combination) and is compared against the
 * rank-preserving threshold r_radius = dist_to_rdist(radius).
 */
#ifndef SKLEARN_PDR_RADIUS_NEIGHBORS_HPP
#define SKLEARN_PDR_RADIUS_NEIGHBORS_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "metric_kernels.hpp"
#include "base.hpp"
#include "datasets_pair.hpp"
#include "heap.hpp"

namespace sklearn {
namespace pdr {

template <typename T>
struct RadiusNeighbors : ReductionHooks {
    DatasetsPair<T> dp;
    idx_t n_samples_X;
    idx_t n_samples_Y;
    double r_radius;       // rank-preserving (surrogate) threshold
    bool sort_results;
    const ChunkingConfig& cfg;

    // Main per-row results.
    std::vector<std::vector<idx_t>> neigh_indices;
    std::vector<std::vector<double>> neigh_distances;
    // Per-thread storage used by parallel_on_Y, merged at the end.
    std::vector<std::vector<std::vector<idx_t>>> tl_indices;
    std::vector<std::vector<std::vector<double>>> tl_distances;
    // Where compute_and_reduce appends for each thread (main or thread-local).
    std::vector<std::vector<std::vector<idx_t>>*> idx_target;
    std::vector<std::vector<std::vector<double>>*> dist_target;

    RadiusNeighbors(DatasetsPair<T> dp_, idx_t n_samples_X_, idx_t n_samples_Y_,
                    double r_radius_, bool sort_results_, const ChunkingConfig& cfg_)
        : dp(dp_), n_samples_X(n_samples_X_), n_samples_Y(n_samples_Y_),
          r_radius(r_radius_), sort_results(sort_results_), cfg(cfg_) {
        neigh_indices.resize(n_samples_X);
        neigh_distances.resize(n_samples_X);
        idx_target.resize(cfg.chunks_n_threads);
        dist_target.resize(cfg.chunks_n_threads);
    }

    void compute_and_reduce(idx_t X_start, idx_t X_end, idx_t Y_start, idx_t Y_end, int tn) {
        auto& IT = *idx_target[tn];
        auto& DT = *dist_target[tn];
        for (idx_t i = X_start; i < X_end; ++i) {
            for (idx_t j = Y_start; j < Y_end; ++j) {
                double rd = dp.rdist(i, j);
                if (rd <= r_radius) {
                    DT[i].push_back(rd);
                    IT[i].push_back(j);
                }
            }
        }
    }

    // ---- parallel_on_X: write straight into the (disjoint) main rows ----
    void parallel_on_X_init_chunk(int tn, idx_t /*X_start*/, idx_t /*X_end*/) {
        idx_target[tn] = &neigh_indices;
        dist_target[tn] = &neigh_distances;
    }
    void parallel_on_X_prange_iter_finalize(int /*tn*/, idx_t X_start, idx_t X_end) {
        if (!sort_results) return;
        for (idx_t i = X_start; i < X_end; ++i)
            simultaneous_sort(neigh_distances[i].data(), neigh_indices[i].data(),
                              static_cast<idx_t>(neigh_indices[i].size()));
    }

    // ---- parallel_on_Y: accumulate per thread, merge at the end ----
    void parallel_on_Y_init() {
        tl_indices.assign(cfg.chunks_n_threads,
                          std::vector<std::vector<idx_t>>(n_samples_X));
        tl_distances.assign(cfg.chunks_n_threads,
                            std::vector<std::vector<double>>(n_samples_X));
        for (idx_t tn = 0; tn < cfg.chunks_n_threads; ++tn) {
            idx_target[tn] = &tl_indices[tn];
            dist_target[tn] = &tl_distances[tn];
        }
    }
    void parallel_on_Y_finalize() {
        #pragma omp parallel for schedule(static) num_threads(cfg.effective_n_threads)
        for (idx_t i = 0; i < n_samples_X; ++i) {
            for (idx_t tn = 0; tn < cfg.chunks_n_threads; ++tn) {
                auto& ti = tl_indices[tn][i];
                auto& td = tl_distances[tn][i];
                neigh_indices[i].insert(neigh_indices[i].end(), ti.begin(), ti.end());
                neigh_distances[i].insert(neigh_distances[i].end(), td.begin(), td.end());
            }
            if (sort_results)
                simultaneous_sort(neigh_distances[i].data(), neigh_indices[i].data(),
                                  static_cast<idx_t>(neigh_indices[i].size()));
        }
    }

    void compute_exact_distances() {
        #pragma omp parallel for schedule(static) num_threads(cfg.effective_n_threads)
        for (idx_t i = 0; i < n_samples_X; ++i)
            for (std::size_t j = 0; j < neigh_indices[i].size(); ++j) {
                double r = std::max(neigh_distances[i][j], 0.0);
                neigh_distances[i][j] = dp.metric->rdist_to_dist(static_cast<T>(r));
            }
    }
};

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_RADIUS_NEIGHBORS_HPP
