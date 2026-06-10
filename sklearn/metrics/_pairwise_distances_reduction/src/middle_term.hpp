/* Euclidean "middle term" via BLAS GEMM, plus squared row norms.
 *
 * The Euclidean specialization decomposes the squared distance as
 *   ||x - y||^2 = ||x||^2 - 2 x.y^T + ||y||^2
 * and computes the -2 x.y^T term for a chunk with one GEMM call. Mirrors
 * _middle_term_computer.pyx.tp: the GEMM is always float64 (float32 inputs are
 * upcast by the caller), so only dgemm is needed.
 *
 * dgemm is obtained at runtime from scipy.linalg.cython_blas.__pyx_capi__, so we
 * use the exact same BLAS scipy is linked against (no second BLAS in-process).
 */
#ifndef SKLEARN_PDR_MIDDLE_TERM_HPP
#define SKLEARN_PDR_MIDDLE_TERM_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>

#include "base.hpp"  // idx_t

namespace sklearn {
namespace pdr {

// scipy's cython_blas uses the Fortran ABI (all-pointer args, non-const).
// NOTE: blas_int is assumed to be `int` (LP64). An ILP64 scipy/MKL build uses
// 64-bit BLAS ints; detecting that (as sklearn's _cython_blas does) is required
// before this backend becomes the default. TODO(cpp-port): blas_int detection.
using dgemm_t = void (*)(char*, char*, int*, int*, int*, double*, double*, int*,
                         double*, int*, double*, double*, int*);

inline dgemm_t load_dgemm() {
    // Cached; first call must hold the GIL (it imports scipy and reads a capsule).
    static dgemm_t ptr = nullptr;
    if (ptr) return ptr;
    namespace nb = nanobind;
    nb::object capi =
        nb::module_::import_("scipy.linalg.cython_blas").attr("__pyx_capi__");
    nb::object cap = capi["dgemm"];
    PyObject* c = cap.ptr();
    void* p = PyCapsule_GetPointer(c, PyCapsule_GetName(c));
    if (p == nullptr)
        throw std::runtime_error("failed to load dgemm from scipy.linalg.cython_blas");
    ptr = reinterpret_cast<dgemm_t>(p);
    return ptr;
}

// C := -2 * A @ B^T, with A (m x k), B (n x k) and C (m x n) all row-major
// float64. Implemented as a single dgemm using the RowMajor convention from
// sklearn.utils._cython_blas._gemm (swap operands + transpose flags).
inline void gemm_minus2_XYt(dgemm_t dgemm, const double* A, const double* B,
                            double* C, int m, int n, int k) {
    char trans_a = 'N';  // op(A) = A
    char trans_b = 'T';  // op(B) = B^T
    int m_ = m, n_ = n, k_ = k;
    int lda = k, ldb = k, ldc = n;
    double alpha = -2.0, beta = 0.0;
    // RowMajor: dgemm(&tb, &ta, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc)
    dgemm(&trans_b, &trans_a, &n_, &m_, &k_, &alpha,
          const_cast<double*>(B), &ldb, const_cast<double*>(A), &lda,
          &beta, C, &ldc);
}

// Squared Euclidean norm of each of the n rows of X (n x d, row-major).
// Accumulated in double regardless of T (matches the Cython upcast behavior).
template <typename T>
inline std::vector<double> squared_row_norms(const T* X, idx_t n, idx_t d,
                                             idx_t n_threads) {
    std::vector<double> norms(n);
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (idx_t i = 0; i < n; ++i) {
        double s = 0.0;
        const T* row = X + i * d;
        for (idx_t j = 0; j < d; ++j) {
            double v = static_cast<double>(row[j]);
            s += v * v;
        }
        norms[i] = s;
    }
    return norms;
}

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_MIDDLE_TERM_HPP
