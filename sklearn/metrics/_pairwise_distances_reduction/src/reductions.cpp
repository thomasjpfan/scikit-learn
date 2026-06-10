/* nanobind bindings for the C++ pairwise-distances reductions.
 *
 * Step 3a: ArgKmin for the Euclidean metric (dense, float32/float64), via the
 * generic per-pair functor path. The Python dispatcher routes here only for
 * supported cases when the C++ backend is enabled (see _dispatcher.py); all
 * other cases still use the Cython implementation.
 */
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "metric_kernels.hpp"
#include "argkmin.hpp"
#include "base.hpp"

namespace nb = nanobind;
using namespace sklearn::pdr;

namespace {

// Wrap a std::vector as a (rows, cols) numpy array that owns the buffer (the
// vector is moved to the heap and freed when the array is garbage-collected).
template <typename E>
nb::object owned_2d(std::vector<E>&& v, std::size_t rows, std::size_t cols) {
    auto* heap = new std::vector<E>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<E>*>(p);
    });
    std::size_t shape[2] = {rows, cols};
    return nb::cast(nb::ndarray<nb::numpy, E>(heap->data(), 2, shape, owner));
}

template <typename T>
nb::object argkmin_compute(
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig> X,
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig> Y,
    idx_t k, idx_t chunk_size, idx_t n_threads, int strategy,
    std::uintptr_t metric_ptr, bool use_squared_distances, bool return_distance) {
    idx_t n_X = static_cast<idx_t>(X.shape(0));
    idx_t n_Y = static_cast<idx_t>(Y.shape(0));
    idx_t n_features = static_cast<idx_t>(X.shape(1));

    // The MetricBase<T> functor is built and owned by a Cython DistanceMetric
    // (the single metric parser), which the dispatcher keeps alive for the
    // duration of this call; we only borrow it here.
    const sklearn::metrics::MetricBase<T>* metric =
        reinterpret_cast<const sklearn::metrics::MetricBase<T>*>(metric_ptr);

    ChunkingConfig cfg = make_chunking_config(
        n_X, n_Y, chunk_size, n_threads, static_cast<Strategy>(strategy));

    ArgKmin<T> red(X.data(), Y.data(), n_features, n_X, n_Y, k,
                   metric, use_squared_distances, cfg);

    {
        nb::gil_scoped_release release;
        run_reduction(red, cfg);
        if (return_distance) red.compute_exact_distances();
    }

    nb::object indices = owned_2d(std::move(red.argkmin_indices),
                                  static_cast<std::size_t>(n_X),
                                  static_cast<std::size_t>(k));
    if (return_distance) {
        nb::object distances = owned_2d(std::move(red.argkmin_distances),
                                        static_cast<std::size_t>(n_X),
                                        static_cast<std::size_t>(k));
        return nb::make_tuple(distances, indices);
    }
    return indices;
}

}  // namespace

NB_MODULE(_reductions, m) {
    m.def("argkmin_compute", &argkmin_compute<double>,
          nb::arg("X"), nb::arg("Y"), nb::arg("k"), nb::arg("chunk_size"),
          nb::arg("n_threads"), nb::arg("strategy"), nb::arg("metric_ptr"),
          nb::arg("use_squared_distances"), nb::arg("return_distance"));
    m.def("argkmin_compute", &argkmin_compute<float>,
          nb::arg("X"), nb::arg("Y"), nb::arg("k"), nb::arg("chunk_size"),
          nb::arg("n_threads"), nb::arg("strategy"), nb::arg("metric_ptr"),
          nb::arg("use_squared_distances"), nb::arg("return_distance"));
}
