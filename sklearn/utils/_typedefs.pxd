#!python
cimport numpy as np

# Floating point/data type
ctypedef np.float64_t DTYPE_t  # WARNING: should match DTYPE in typedefs.pyx
ctypedef np.float32_t FTYPE_t  # WARNING: should match DTYPE in typedefs.pyx

cdef enum:
    DTYPECODE = np.NPY_FLOAT64
    FTYPECODE = np.NPY_FLOAT32
    ITYPECODE = np.NPY_INTP
    INT32TYPECODE = np.NPY_INT32
    INT64TYPECODE = np.NPY_INT64

# Index/integer type.
#  WARNING: ITYPE_t must be a signed integer type or you will have a bad time!
ctypedef np.intp_t ITYPE_t  # WARNING: should match ITYPE in typedefs.pyx
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t
ctypedef np.longlong_t LONGLONGTYPE_t  # WARNING: should match DTYPE in typedefs.pyx

# Fused type for certain operations
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t
