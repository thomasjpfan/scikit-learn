/* Chunking configuration and the parallel-on-X / parallel-on-Y skeletons shared
 * by the pairwise-distances reductions.
 *
 * This is the C++ counterpart of BaseDistancesReduction{32,64} in _base.pyx.tp:
 * it computes the chunk layout + parallelization strategy, then drives the two
 * nested-chunk OpenMP loops, calling hook methods on a concrete reduction type.
 *
 * Reductions provide their behavior by defining hook methods (see ReductionHooks
 * for the full set and their no-op defaults). The skeleton is templated on the
 * concrete reduction type, so hook calls are resolved at compile time (no
 * virtual dispatch in the chunk loops).
 */
#ifndef SKLEARN_PDR_BASE_HPP
#define SKLEARN_PDR_BASE_HPP

#include <algorithm>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sklearn {
namespace pdr {

using idx_t = std::intptr_t;

enum class Strategy { Auto = 0, ParallelOnX = 1, ParallelOnY = 2 };

struct ChunkingConfig {
    idx_t chunk_size;
    idx_t effective_n_threads;
    idx_t chunks_n_threads;

    idx_t n_samples_X, X_n_samples_chunk, X_n_chunks, X_n_samples_last_chunk;
    idx_t n_samples_Y, Y_n_samples_chunk, Y_n_chunks, Y_n_samples_last_chunk;

    bool execute_in_parallel_on_Y;

    idx_t X_chunk_start(idx_t c) const { return c * X_n_samples_chunk; }
    idx_t X_chunk_end(idx_t c) const {
        return X_chunk_start(c) + (c == X_n_chunks - 1 ? X_n_samples_last_chunk : X_n_samples_chunk);
    }
    idx_t Y_chunk_start(idx_t c) const { return c * Y_n_samples_chunk; }
    idx_t Y_chunk_end(idx_t c) const {
        return Y_chunk_start(c) + (c == Y_n_chunks - 1 ? Y_n_samples_last_chunk : Y_n_samples_chunk);
    }
};

inline ChunkingConfig make_chunking_config(
    idx_t n_samples_X, idx_t n_samples_Y,
    idx_t chunk_size, idx_t effective_n_threads, Strategy strategy) {
    ChunkingConfig c;
    c.chunk_size = chunk_size;
    c.effective_n_threads = effective_n_threads;

    c.n_samples_X = n_samples_X;
    c.X_n_samples_chunk = std::min(n_samples_X, chunk_size);
    idx_t X_n_full = n_samples_X / c.X_n_samples_chunk;
    idx_t X_rem = n_samples_X % c.X_n_samples_chunk;
    c.X_n_chunks = X_n_full + (X_rem != 0 ? 1 : 0);
    c.X_n_samples_last_chunk = X_rem != 0 ? X_rem : c.X_n_samples_chunk;

    c.n_samples_Y = n_samples_Y;
    c.Y_n_samples_chunk = std::min(n_samples_Y, chunk_size);
    idx_t Y_n_full = n_samples_Y / c.Y_n_samples_chunk;
    idx_t Y_rem = n_samples_Y % c.Y_n_samples_chunk;
    c.Y_n_chunks = Y_n_full + (Y_rem != 0 ? 1 : 0);
    c.Y_n_samples_last_chunk = Y_rem != 0 ? Y_rem : c.Y_n_samples_chunk;

    if (strategy == Strategy::Auto) {
        // Same heuristic as BaseDistancesReduction.__init__.
        if (n_samples_Y < n_samples_X)
            strategy = Strategy::ParallelOnX;
        else if (4 * chunk_size * effective_n_threads < n_samples_X)
            strategy = Strategy::ParallelOnX;
        else
            strategy = Strategy::ParallelOnY;
    }
    c.execute_in_parallel_on_Y = (strategy == Strategy::ParallelOnY);
    c.chunks_n_threads = std::min(
        c.execute_in_parallel_on_Y ? c.Y_n_chunks : c.X_n_chunks,
        effective_n_threads);
    return c;
}

inline int thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Default no-op hooks. Reductions inherit this and override only what they need.
struct ReductionHooks {
    void compute_and_reduce(idx_t, idx_t, idx_t, idx_t, int) {}
    void parallel_on_X_parallel_init(int) {}
    void parallel_on_X_init_chunk(int, idx_t, idx_t) {}
    void parallel_on_X_pre_compute(idx_t, idx_t, idx_t, idx_t, int) {}
    void parallel_on_X_prange_iter_finalize(int, idx_t, idx_t) {}
    void parallel_on_X_parallel_finalize(int) {}
    void parallel_on_Y_init() {}
    void parallel_on_Y_parallel_init(int, idx_t, idx_t) {}
    void parallel_on_Y_pre_compute(idx_t, idx_t, idx_t, idx_t, int) {}
    void parallel_on_Y_synchronize(idx_t, idx_t) {}
    void parallel_on_Y_finalize() {}
};

template <typename R>
void parallel_on_X(R& r, const ChunkingConfig& c) {
    #pragma omp parallel num_threads(c.chunks_n_threads)
    {
        int tn = thread_num();
        r.parallel_on_X_parallel_init(tn);

        #pragma omp for schedule(static)
        for (idx_t xc = 0; xc < c.X_n_chunks; ++xc) {
            idx_t X_start = c.X_chunk_start(xc), X_end = c.X_chunk_end(xc);
            r.parallel_on_X_init_chunk(tn, X_start, X_end);
            for (idx_t yc = 0; yc < c.Y_n_chunks; ++yc) {
                idx_t Y_start = c.Y_chunk_start(yc), Y_end = c.Y_chunk_end(yc);
                r.parallel_on_X_pre_compute(X_start, X_end, Y_start, Y_end, tn);
                r.compute_and_reduce(X_start, X_end, Y_start, Y_end, tn);
            }
            r.parallel_on_X_prange_iter_finalize(tn, X_start, X_end);
        }
        r.parallel_on_X_parallel_finalize(tn);
    }
}

template <typename R>
void parallel_on_Y(R& r, const ChunkingConfig& c) {
    r.parallel_on_Y_init();

    for (idx_t xc = 0; xc < c.X_n_chunks; ++xc) {
        idx_t X_start = c.X_chunk_start(xc), X_end = c.X_chunk_end(xc);

        #pragma omp parallel num_threads(c.chunks_n_threads)
        {
            int tn = thread_num();
            r.parallel_on_Y_parallel_init(tn, X_start, X_end);

            #pragma omp for schedule(static)
            for (idx_t yc = 0; yc < c.Y_n_chunks; ++yc) {
                idx_t Y_start = c.Y_chunk_start(yc), Y_end = c.Y_chunk_end(yc);
                r.parallel_on_Y_pre_compute(X_start, X_end, Y_start, Y_end, tn);
                r.compute_and_reduce(X_start, X_end, Y_start, Y_end, tn);
            }
        }
        r.parallel_on_Y_synchronize(X_start, X_end);
    }
    r.parallel_on_Y_finalize();
}

template <typename R>
void run_reduction(R& r, const ChunkingConfig& c) {
    if (c.execute_in_parallel_on_Y)
        parallel_on_Y(r, c);
    else
        parallel_on_X(r, c);
}

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_BASE_HPP
