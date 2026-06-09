/* Temporary validation module for metric_kernels.hpp (commit 2a of the
 * C++/nanobind port). It exposes thin wrappers that run the C++ metric functors
 * over dense and CSR inputs so a Python test can compare them against the
 * current Cython DistanceMetric (the behavior we must preserve). Removed in the
 * commit that wires _dist_metrics to delegate to the kernels.
 */
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "metric_kernels.hpp"

namespace nb = nanobind;
using namespace sklearn::metrics;

using f64_2d = nb::ndarray<const double, nb::ndim<2>, nb::c_contig>;
using f64_1d = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using i32_1d = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>;
using out_2d = nb::ndarray<double, nb::ndim<2>, nb::c_contig>;

// Build a double-precision functor from a metric name + the union of possible
// parameters (unused ones are passed as empty arrays / ignored).
static std::unique_ptr<MetricBase<double>> make_metric(
        const std::string& name, double p,
        const double* V, std::intptr_t Vn,
        const double* VI, std::intptr_t VIn,
        const double* w, std::intptr_t wn) {
    MetricBase<double>* m = nullptr;
    if (name == "euclidean") m = make_euclidean<double>();
    else if (name == "seuclidean") m = make_seuclidean<double>(V, Vn);
    else if (name == "manhattan") m = make_manhattan<double>();
    else if (name == "chebyshev") m = make_chebyshev<double>();
    else if (name == "minkowski") m = make_minkowski<double>(p, wn ? w : nullptr, wn);
    else if (name == "mahalanobis") m = make_mahalanobis<double>(VI, VIn);
    else if (name == "hamming") m = make_hamming<double>();
    else if (name == "canberra") m = make_canberra<double>();
    else if (name == "braycurtis") m = make_braycurtis<double>();
    else if (name == "jaccard") m = make_jaccard<double>();
    else if (name == "matching") m = make_matching<double>();
    else if (name == "dice") m = make_dice<double>();
    else if (name == "kulsinski") m = make_kulsinski<double>();
    else if (name == "rogerstanimoto") m = make_rogerstanimoto<double>();
    else if (name == "russellrao") m = make_russellrao<double>();
    else if (name == "sokalmichener") m = make_sokalmichener<double>();
    else if (name == "sokalsneath") m = make_sokalsneath<double>();
    else if (name == "haversine") m = make_haversine<double>();
    else throw std::invalid_argument("unknown metric: " + name);
    return std::unique_ptr<MetricBase<double>>(m);
}

// D[i, j] = dist(X[i], Y[j]) using the dense functor interface.
static void pairwise_dense(const std::string& name, f64_2d X, f64_2d Y, out_2d D,
                           double p, f64_1d V, f64_1d VI_flat, std::intptr_t VIn, f64_1d w) {
    auto m = make_metric(name, p, V.data(), static_cast<std::intptr_t>(V.shape(0)),
                         VI_flat.data(), VIn, w.data(), static_cast<std::intptr_t>(w.shape(0)));
    std::intptr_t n1 = X.shape(0), n2 = Y.shape(0), d = X.shape(1);
    const double* xp = X.data();
    const double* yp = Y.data();
    double* dp = D.data();
    for (std::intptr_t i = 0; i < n1; ++i)
        for (std::intptr_t j = 0; j < n2; ++j)
            dp[i * n2 + j] = m->dist(xp + i * d, yp + j * d, d);
}

// D[i, j] = dist_csr(X[i], Y[j]) using the CSR functor interface.
static void pairwise_csr(const std::string& name,
                         f64_1d Xdata, i32_1d Xindices, i32_1d Xindptr,
                         f64_1d Ydata, i32_1d Yindices, i32_1d Yindptr,
                         std::intptr_t n_features, out_2d D,
                         double p, f64_1d V, f64_1d VI_flat, std::intptr_t VIn, f64_1d w) {
    auto m = make_metric(name, p, V.data(), static_cast<std::intptr_t>(V.shape(0)),
                         VI_flat.data(), VIn, w.data(), static_cast<std::intptr_t>(w.shape(0)));
    std::intptr_t n1 = static_cast<std::intptr_t>(Xindptr.shape(0)) - 1;
    std::intptr_t n2 = static_cast<std::intptr_t>(Yindptr.shape(0)) - 1;
    const double* xd = Xdata.data(); const std::int32_t* xi = Xindices.data(); const std::int32_t* xp = Xindptr.data();
    const double* yd = Ydata.data(); const std::int32_t* yi = Yindices.data(); const std::int32_t* yp = Yindptr.data();
    double* dp = D.data();
    for (std::intptr_t i = 0; i < n1; ++i)
        for (std::intptr_t j = 0; j < n2; ++j)
            dp[i * n2 + j] = m->dist_csr(xd, xi, yd, yi, xp[i], xp[i + 1], yp[j], yp[j + 1], n_features);
}

NB_MODULE(_metric_kernels_test, m) {
    m.def("pairwise_dense", &pairwise_dense);
    m.def("pairwise_csr", &pairwise_csr);
}
