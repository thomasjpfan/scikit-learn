/* Fixed-size max-heap push and simultaneous (values, indices) sort.
 *
 * Faithful C++ port of sklearn/utils/_heap.pyx (heap_push) and
 * sklearn/utils/_sorting.pyx (simultaneous_sort). The sort is an introsort and
 * is NOT stable -- this matches the Cython implementation and is the reason the
 * discrete metrics (hamming, the boolean metrics) are excluded from the
 * pairwise-distances reductions.
 *
 * Templated on the value type V (the reductions use double heaps regardless of
 * input dtype) and a fixed intptr_t index type.
 */
#ifndef SKLEARN_PDR_HEAP_HPP
#define SKLEARN_PDR_HEAP_HPP

#include <cmath>
#include <cstdint>

namespace sklearn {
namespace pdr {

using idx_t = std::intptr_t;

// Push (val, val_idx) onto the fixed-size max-heap stored as parallel arrays
// (values, indices) of length `size`. No-op if val is not smaller than the
// current max (values[0]).
template <typename V>
inline void heap_push(V* values, idx_t* indices, idx_t size, V val, idx_t val_idx) {
    if (val >= values[0]) return;

    values[0] = val;
    indices[0] = val_idx;

    idx_t current_idx = 0;
    while (true) {
        idx_t left_child_idx = 2 * current_idx + 1;
        idx_t right_child_idx = left_child_idx + 1;
        idx_t swap_idx;

        if (left_child_idx >= size) {
            break;
        } else if (right_child_idx >= size) {
            if (values[left_child_idx] > val)
                swap_idx = left_child_idx;
            else
                break;
        } else if (values[left_child_idx] >= values[right_child_idx]) {
            if (val < values[left_child_idx])
                swap_idx = left_child_idx;
            else
                break;
        } else {
            if (val < values[right_child_idx])
                swap_idx = right_child_idx;
            else
                break;
        }

        values[current_idx] = values[swap_idx];
        indices[current_idx] = indices[swap_idx];
        current_idx = swap_idx;
    }

    values[current_idx] = val;
    indices[current_idx] = val_idx;
}

namespace detail {

template <typename V>
inline void swap(V* values, idx_t* indices, idx_t i, idx_t j) {
    V tv = values[i]; values[i] = values[j]; values[j] = tv;
    idx_t ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
}

template <typename V>
inline void sift_down(V* values, idx_t* indices, idx_t start, idx_t end) {
    idx_t root = start;
    while (true) {
        idx_t child = root * 2 + 1;
        idx_t maxind = root;
        if (child < end && values[maxind] < values[child]) maxind = child;
        if (child + 1 < end && values[maxind] < values[child + 1]) maxind = child + 1;
        if (maxind == root) break;
        swap(values, indices, root, maxind);
        root = maxind;
    }
}

template <typename V>
inline void heapsort(V* values, idx_t* indices, idx_t n) {
    idx_t start = (n - 2) / 2;
    idx_t end = n;
    while (true) {
        sift_down(values, indices, start, end);
        if (start == 0) break;
        --start;
    }
    end = n - 1;
    while (end > 0) {
        swap(values, indices, 0, end);
        sift_down(values, indices, 0, end);
        --end;
    }
}

template <typename V>
inline void insertion_sort(V* values, idx_t* indices, idx_t n) {
    for (idx_t i = 1; i < n; ++i) {
        V temp_val = values[i];
        idx_t temp_idx = indices[i];
        idx_t j = i;
        while (j > 0 && values[j - 1] > temp_val) {
            values[j] = values[j - 1];
            indices[j] = indices[j - 1];
            --j;
        }
        values[j] = temp_val;
        indices[j] = temp_idx;
    }
}

template <typename V>
inline V inplace_median3(V* values, idx_t* indices, idx_t n) {
    idx_t pivot_idx = n / 2;
    if (values[0] > values[n - 1]) swap(values, indices, 0, n - 1);
    if (values[n - 1] > values[pivot_idx]) {
        swap(values, indices, n - 1, pivot_idx);
        if (values[0] > values[n - 1]) swap(values, indices, 0, n - 1);
    }
    return values[n - 1];
}

template <typename V>
inline V median3(const V* values, idx_t n) {
    V a = values[0], b = values[n / 2], c = values[n - 1];
    if (a < b) {
        if (b < c) return b;
        else if (a < c) return c;
        else return a;
    } else if (b < c) {
        if (a < c) return a;
        else return c;
    } else {
        return b;
    }
}

template <typename V>
inline void introsort_2way(V* values, idx_t* indices, idx_t n, idx_t maxd) {
    while (n > 15) {
        if (maxd <= 0) { heapsort(values, indices, n); return; }
        --maxd;

        V pivot = inplace_median3(values, indices, n);
        idx_t i = 1;      // median3 ensures values[0] <= pivot
        idx_t j = n - 2;  // median3 ensures values[-1] >= pivot
        while (true) {
            while (i <= j && values[i] < pivot) ++i;
            while (i <= j && values[j] > pivot) --j;
            if (i >= j) break;
            swap(values, indices, i, j);
            ++i; --j;
        }
        idx_t pivot_idx = i;
        swap(values, indices, pivot_idx, n - 1);

        introsort_2way(values, indices, pivot_idx, maxd);
        values += pivot_idx + 1;
        indices += pivot_idx + 1;
        n -= pivot_idx + 1;
    }
    insertion_sort(values, indices, n);
}

template <typename V>
inline void introsort_3way(V* values, idx_t* indices, idx_t n, idx_t maxd) {
    while (n > 15) {
        if (maxd <= 0) { heapsort(values, indices, n); return; }
        --maxd;

        V pivot = median3(values, n);
        idx_t i = 0, l = 0, r = n;
        while (i < r) {
            if (values[i] < pivot) { swap(values, indices, i, l); ++i; ++l; }
            else if (values[i] > pivot) { --r; swap(values, indices, i, r); }
            else { ++i; }
        }
        introsort_3way(values, indices, l, maxd);
        values += r;
        indices += r;
        n -= r;
    }
    insertion_sort(values, indices, n);
}

}  // namespace detail

// Sort (values, indices) in ascending order of values. Introsort; not stable.
template <typename V>
inline void simultaneous_sort(V* values, idx_t* indices, idx_t n,
                              bool use_three_way_partition = false) {
    if (n == 0) return;
    idx_t maxd = 2 * static_cast<idx_t>(std::log2(static_cast<double>(n)));
    if (use_three_way_partition)
        detail::introsort_3way(values, indices, n, maxd);
    else
        detail::introsort_2way(values, indices, n, maxd);
}

}  // namespace pdr
}  // namespace sklearn

#endif  // SKLEARN_PDR_HEAP_HPP
