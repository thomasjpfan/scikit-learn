/* Toolchain smoke test for the nanobind build pipeline.
 *
 * This is a temporary module introduced in the first commit of the
 * Cython -> C++/nanobind port of _pairwise_distances_reduction. Its only job is
 * to prove, across the full CI / wheel matrix, that:
 *   - nanobind is found and its support library links,
 *   - the C++17 standard is actually applied (override_options), and
 *   - the nb::ndarray numpy boundary compiles and round-trips.
 * It is removed once the real reductions module lands.
 */
#include <cstddef>
#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Sum of a 1-D, C-contiguous float64 array. Exercises the nb::ndarray boundary
// with the same dtype/contiguity constraints the real module will rely on.
static double sum1d(nb::ndarray<const double, nb::ndim<1>, nb::c_contig> a) {
    double total = 0.0;
    for (std::size_t i = 0; i < a.shape(0); ++i) {
        total += a(i);
    }
    return total;
}

NB_MODULE(_nanobind_smoke, m) {
    m.def("add", [](std::int64_t a, std::int64_t b) { return a + b; });
    m.def("sum1d", &sum1d);
    // Lets the Python-side test assert C++17 (or newer) was used: 201703L.
    m.attr("cplusplus") = static_cast<std::int64_t>(__cplusplus);
}
