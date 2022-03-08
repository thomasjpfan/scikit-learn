#!python
cimport numpy as np

# Floating point/data type
ctypedef np.float64_t DTYPE_t  # WARNING: should match DTYPE in typedefs.pyx
ctypedef np.float32_t FTYPE_t  # WARNING: should match DTYPE in typedefs.pyx

cdef enum:
    DTYPECODE = np.NPY_FLOAT64
    FTYPECODE = np.NPY_FLOAT32
    ITYPECODE = np.NPY_INTP
    INT64TYPECODE = np.NPY_INT64

# Index/integer type.
#  WARNING: ITYPE_t must be a signed integer type or you will have a bad time!
ctypedef np.intp_t ITYPE_t  # WARNING: should match ITYPE in typedefs.pyx
ctypedef np.int64_t INT64_t

# Fused type for certain operations
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t
