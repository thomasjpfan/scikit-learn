from cython cimport floating
from libc.math cimport log2

cdef inline void dual_swap(
    floating* darr,
    ITYPE_t *iarr,
    ITYPE_t a,
    ITYPE_t b,
) nogil:
    """Swap the values at index a and b of both darr and iarr"""
    cdef floating dtmp = darr[a]
    darr[a] = darr[b]
    darr[b] = dtmp

    cdef ITYPE_t itmp = iarr[a]
    iarr[a] = iarr[b]
    iarr[b] = itmp


cdef inline void sift_down(
    floating* values,
    ITYPE_t* samples,
    ITYPE_t start,
    ITYPE_t end,
) nogil:
    # Restore heap order in values[start:end] by moving the max element to start.
    cdef ITYPE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and values[maxind] < values[child]:
            maxind = child
        if child + 1 < end and values[maxind] < values[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            dual_swap(values, samples, root, maxind)
            root = maxind


cdef inline void heapsort(
    floating* values,
    ITYPE_t* samples,
    ITYPE_t n
) nogil:
    cdef:
        ITYPE_t start = (n - 2) / 2
        ITYPE_t end = n

     # heapify
    while True:
        sift_down(values, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        dual_swap(values, samples, 0, end)
        sift_down(values, samples, 0, end)
        end = end - 1


cdef inline int simultaneous_sort(
    floating* values,
    ITYPE_t* indices,
    ITYPE_t size,
    bint use_introsort=0,
) nogil:
    """
    Perform a recursive quicksort on the values array as to sort them ascendingly.
    This simultaneously performs the swaps on both the values and the indices arrays.

    The numpy equivalent is:

        def simultaneous_sort(dist, idx):
             i = np.argsort(dist)
             return dist[i], idx[i]

    If use_introsort=1, then the introsort algorithm is used. This sorting algorithm
    switches from quicksort to heapsort when the recursion depth based on
    based on 2 * log2(size).

    Notes
    -----
    Arrays are manipulated via a pointer to there first element and their size
    as to ease the processing of dynamically allocated buffers.
    """
    # TODO: In order to support discrete distance metrics, we need to have a
    # simultaneous sort which breaks ties on indices when distances are identical.
    # The best might be using a std::stable_sort and a Comparator which might need
    # an Array of Structures (AoS) instead of the Structure of Arrays (SoA)
    # currently used.
    if use_introsort == 1:
        _simultaneous_sort(values, indices, size, 2 * <int>log2(size), 1)
    else:
        _simultaneous_sort(values, indices, size, -1, 0)


cdef inline floating median3(floating* values, ITYPE_t size) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef floating a = values[0], b = values[size / 2], c = values[size - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


cdef inline int _simultaneous_sort(
    floating* values,
    ITYPE_t* indices,
    ITYPE_t size,
    int max_depth,
    bint use_introsort,
) nogil:
    cdef:
        ITYPE_t l, r, i
        floating pivot_val

    # in the small-array case, do things efficiently
    if size <= 1:
        pass
    elif size == 2:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
    elif size == 3:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
        if values[1] > values[2]:
            dual_swap(values, indices, 1, 2)
            if values[0] > values[1]:
                dual_swap(values, indices, 0, 1)
    elif use_introsort and max_depth <= 0:
        heapsort(values, indices, size)
    else:
        # Determine the pivot using the median-of-three rule.
        # The smallest of the three is moved to the beginning of the array,
        # the middle (the pivot value) is moved to the end, and the largest
        # is moved to the pivot index.
        pivot_val = median3(values, size)

        # Three-way partition indices about pivot. At the end of this operation,
        # pivot_idx will contain the pivot value, everything to the left will be
        # smaller, and everything to the right will be larger.
        i = l = 0
        r = size
        while i < r:
            if values[i] < pivot_val:
                dual_swap(values, indices, i, l)
                i += 1
                l += 1
            elif values[i] > pivot_val:
                r -= 1
                dual_swap(values, indices, i, r)
            else:
                i += 1

        # Recursively sort each side of the pivot
        if l > 1:
            _simultaneous_sort(values, indices, l, max_depth - 1, use_introsort)
        if l + 2 < size:
            _simultaneous_sort(values + l + 1,
                               indices + l + 1,
                               size - l - 1, max_depth - 1, use_introsort)
    return 0
