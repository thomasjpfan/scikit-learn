cimport numpy as np

from libcpp.vector cimport vector
from ..utils._typedefs cimport ITYPE_t, DTYPE_t, INT64_t

ctypedef fused vector_typed:
    vector[DTYPE_t]
    vector[ITYPE_t]
    vector[INT64_t]

cdef np.ndarray vector_to_nd_array(vector_typed * vect_ptr)
