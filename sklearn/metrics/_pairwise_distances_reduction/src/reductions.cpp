/* nanobind bindings for the C++ pairwise-distances reductions.
 *
 * ArgKmin for the dense and CSR cases (float32/float64), via the generic
 * per-pair functor path. The MetricBase<T> functor is built and owned by a
 * Cython DistanceMetric (the single metric parser); the dispatcher passes its
 * address here and keeps it alive for the call. The Python dispatcher routes
 * only supported cases to this backend when it is enabled.
 */
#include <cstdint>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "metric_kernels.hpp"
#include "argkmin.hpp"
#include "base.hpp"
#include "datasets_pair.hpp"

namespace nb = nanobind;
using namespace sklearn::pdr;

namespace {

template <typename T>
using f2d = nb::ndarray<const T, nb::ndim<2>, nb::c_contig>;
template <typename T>
using f1d = nb::ndarray<const T, nb::ndim<1>, nb::c_contig>;
using i1d = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>;

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

// Shared core: run ArgKmin over the (already built) DatasetsPair and return the
// indices (and optionally distances) as numpy arrays.
template <typename T>
nb::object run_argkmin(DatasetsPair<T> dp, idx_t n_X, idx_t n_Y, idx_t k,
                       idx_t chunk_size, idx_t n_threads, int strategy,
                       bool use_squared_distances, bool return_distance) {
    ChunkingConfig cfg = make_chunking_config(
        n_X, n_Y, chunk_size, n_threads, static_cast<Strategy>(strategy));
    ArgKmin<T> red(dp, n_X, n_Y, k, use_squared_distances, cfg);
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

template <typename T>
const sklearn::metrics::MetricBase<T>* as_metric(std::uintptr_t p) {
    return reinterpret_cast<const sklearn::metrics::MetricBase<T>*>(p);
}

template <typename T>
nb::object argkmin_dense_dense(
    f2d<T> X, f2d<T> Y, idx_t k, idx_t chunk_size, idx_t n_threads, int strategy,
    std::uintptr_t metric_ptr, bool use_squared_distances, bool return_distance) {
    idx_t n_X = static_cast<idx_t>(X.shape(0));
    idx_t n_Y = static_cast<idx_t>(Y.shape(0));
    DatasetsPair<T> dp;
    dp.kind = DatasetsPair<T>::DenseDense;
    dp.metric = as_metric<T>(metric_ptr);
    dp.n_features = static_cast<idx_t>(X.shape(1));
    dp.X = X.data();
    dp.Y = Y.data();
    return run_argkmin<T>(dp, n_X, n_Y, k, chunk_size, n_threads, strategy,
                          use_squared_distances, return_distance);
}

template <typename T>
nb::object argkmin_sparse_sparse(
    f1d<T> X_data, i1d X_indices, i1d X_indptr,
    f1d<T> Y_data, i1d Y_indices, i1d Y_indptr, idx_t n_features,
    idx_t k, idx_t chunk_size, idx_t n_threads, int strategy,
    std::uintptr_t metric_ptr, bool use_squared_distances, bool return_distance) {
    idx_t n_X = static_cast<idx_t>(X_indptr.shape(0)) - 1;
    idx_t n_Y = static_cast<idx_t>(Y_indptr.shape(0)) - 1;
    DatasetsPair<T> dp;
    dp.kind = DatasetsPair<T>::SparseSparse;
    dp.metric = as_metric<T>(metric_ptr);
    dp.n_features = n_features;
    dp.X_data = X_data.data();
    dp.X_indices = X_indices.data();
    dp.X_indptr = X_indptr.data();
    dp.Y_data = Y_data.data();
    dp.Y_indices = Y_indices.data();
    dp.Y_indptr = Y_indptr.data();
    return run_argkmin<T>(dp, n_X, n_Y, k, chunk_size, n_threads, strategy,
                          use_squared_distances, return_distance);
}

template <typename T>
nb::object argkmin_sparse_dense(
    f1d<T> X_data, i1d X_indices, i1d X_indptr, f2d<T> Y,
    idx_t k, idx_t chunk_size, idx_t n_threads, int strategy,
    std::uintptr_t metric_ptr, bool use_squared_distances, bool return_distance) {
    idx_t n_X = static_cast<idx_t>(X_indptr.shape(0)) - 1;
    idx_t n_Y = static_cast<idx_t>(Y.shape(0));
    idx_t n_features = static_cast<idx_t>(Y.shape(1));
    std::vector<std::int32_t> dense_indices(n_features);
    for (idx_t c = 0; c < n_features; ++c) dense_indices[c] = static_cast<std::int32_t>(c);
    DatasetsPair<T> dp;
    dp.kind = DatasetsPair<T>::SparseDense;
    dp.metric = as_metric<T>(metric_ptr);
    dp.n_features = n_features;
    dp.X_data = X_data.data();
    dp.X_indices = X_indices.data();
    dp.X_indptr = X_indptr.data();
    dp.Y = Y.data();
    dp.dense_indices = dense_indices.data();
    return run_argkmin<T>(dp, n_X, n_Y, k, chunk_size, n_threads, strategy,
                          use_squared_distances, return_distance);
}

template <typename T>
nb::object argkmin_dense_sparse(
    f2d<T> X, f1d<T> Y_data, i1d Y_indices, i1d Y_indptr,
    idx_t k, idx_t chunk_size, idx_t n_threads, int strategy,
    std::uintptr_t metric_ptr, bool use_squared_distances, bool return_distance) {
    idx_t n_X = static_cast<idx_t>(X.shape(0));
    idx_t n_Y = static_cast<idx_t>(Y_indptr.shape(0)) - 1;
    idx_t n_features = static_cast<idx_t>(X.shape(1));
    std::vector<std::int32_t> dense_indices(n_features);
    for (idx_t c = 0; c < n_features; ++c) dense_indices[c] = static_cast<std::int32_t>(c);
    DatasetsPair<T> dp;
    dp.kind = DatasetsPair<T>::DenseSparse;
    dp.metric = as_metric<T>(metric_ptr);
    dp.n_features = n_features;
    dp.X = X.data();
    dp.Y_data = Y_data.data();
    dp.Y_indices = Y_indices.data();
    dp.Y_indptr = Y_indptr.data();
    dp.dense_indices = dense_indices.data();
    return run_argkmin<T>(dp, n_X, n_Y, k, chunk_size, n_threads, strategy,
                          use_squared_distances, return_distance);
}

}  // namespace

NB_MODULE(_reductions, m) {
    m.def("argkmin_dense_dense", &argkmin_dense_dense<double>);
    m.def("argkmin_dense_dense", &argkmin_dense_dense<float>);
    m.def("argkmin_sparse_sparse", &argkmin_sparse_sparse<double>);
    m.def("argkmin_sparse_sparse", &argkmin_sparse_sparse<float>);
    m.def("argkmin_sparse_dense", &argkmin_sparse_dense<double>);
    m.def("argkmin_sparse_dense", &argkmin_sparse_dense<float>);
    m.def("argkmin_dense_sparse", &argkmin_dense_sparse<double>);
    m.def("argkmin_dense_sparse", &argkmin_dense_sparse<float>);
}
