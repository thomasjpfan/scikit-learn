# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
from .common cimport X_BITSET_DTYPE_C
from .common cimport X_BINNED_DTYPE_C


cdef inline void init_bitset(X_BITSET_DTYPE_C bitset) nogil: # OUT
    cdef:
        unsigned int i
    for i in range(4):
        bitset[i] = 0

cdef inline void insert_bitset(X_BINNED_DTYPE_C val,
                               X_BITSET_DTYPE_C bitset) nogil: # OUT
    cdef:
        X_BINNED_DTYPE_C i1 = val / 64
        X_BINNED_DTYPE_C i2 = val % 64

    # It is assumed that i1 <= 3
    bitset[i1] |= (1 << i2)


cdef inline unsigned char in_bitset(X_BINNED_DTYPE_C val,
                                    X_BITSET_DTYPE_C bitset) nogil:
    cdef:
        X_BINNED_DTYPE_C i1 = val / 64
        X_BINNED_DTYPE_C i2 = val % 64

    if i1 >= 4:
        return 0
    return (bitset[i1] >> i2) & 1
