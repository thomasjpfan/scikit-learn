/* DatasetsPair: rank-preserving distance between a row of X and a row of Y for
 * any dense/CSR combination, evaluated through a borrowed MetricBase<T> functor.
 *
 * C++ counterpart of DatasetsPair{32,64} in _datasets_pair.pyx.tp. The sparse
 * cases reuse the metric's CSR routines; the sparse-dense and dense-sparse cases
 * use the same "pseudo-CSR" trick (the dense row addressed with an
 * arange(n_features) index vector), mirroring SparseDenseDatasetsPair, including
 * the argument swap for the dense-sparse case.
 */
#ifndef SKLEARN_PDR_DATASETS_PAIR_HPP
#define SKLEARN_PDR_DATASETS_PAIR_HPP

#include <cstdint>

#include "metric_kernels.hpp"
#include "base.hpp"  // idx_t

namespace sklearn {
namespace pdr {

using sklearn::metrics::MetricBase;

template <typename T>
struct DatasetsPair {
    enum Kind { DenseDense, SparseSparse, SparseDense, DenseSparse };

    Kind kind = DenseDense;
    const MetricBase<T>* metric = nullptr;  // borrowed
    idx_t n_features = 0;

    // Dense operands (X for dense-X kinds, Y for dense-Y kinds).
    const T* X = nullptr;
    const T* Y = nullptr;

    // CSR operands.
    const T* X_data = nullptr;
    const std::int32_t* X_indices = nullptr;
    const std::int32_t* X_indptr = nullptr;
    const T* Y_data = nullptr;
    const std::int32_t* Y_indices = nullptr;
    const std::int32_t* Y_indptr = nullptr;

    // arange(n_features) used as the column indices of a pseudo-CSR dense row.
    const std::int32_t* dense_indices = nullptr;

    inline double rdist(idx_t i, idx_t j) const {
        switch (kind) {
            case DenseDense:
                return static_cast<double>(
                    metric->rdist(X + i * n_features, Y + j * n_features, n_features));
            case SparseSparse:
                return static_cast<double>(metric->rdist_csr(
                    X_data, X_indices, Y_data, Y_indices,
                    X_indptr[i], X_indptr[i + 1], Y_indptr[j], Y_indptr[j + 1],
                    n_features));
            case SparseDense:
                // X sparse (row i), Y dense (row j) as pseudo-CSR.
                return static_cast<double>(metric->rdist_csr(
                    X_data, X_indices,
                    Y + j * n_features, dense_indices,
                    X_indptr[i], X_indptr[i + 1],
                    0, static_cast<std::int32_t>(n_features), n_features));
            case DenseSparse:
                // X dense (row i), Y sparse (row j). Mirror the Cython swap:
                // sparse Y is x1, dense X (pseudo-CSR) is x2.
                return static_cast<double>(metric->rdist_csr(
                    Y_data, Y_indices,
                    X + i * n_features, dense_indices,
                    Y_indptr[j], Y_indptr[j + 1],
                    0, static_cast<std::int32_t>(n_features), n_features));
        }
        return 0.0;  // unreachable
    }
};

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_DATASETS_PAIR_HPP
