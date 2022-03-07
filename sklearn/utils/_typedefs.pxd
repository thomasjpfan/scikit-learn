#!python
cimport numpy as np

# Floating point/data type
ctypedef np.float64_t DTYPE_t  # WARNING: should match DTYPE in typedefs.pyx
ctypedef np.float32_t FTYPE_t  # WARNING: should match DTYPE in typedefs.pyx

cdef enum:
    DTYPECODE = np.NPY_FLOAT64
    FTYPECODE = np.NPY_FLOAT32
    ITYPECODE = np.NPY_INTP
    LONGLONGTYPECODE = np.NPY_LONGLONG

# Index/integer type.
#  WARNING: ITYPE_t must be a signed integer type or you will have a bad time!
ctypedef np.intp_t ITYPE_t  # WARNING: should match ITYPE in typedefs.pyx
ctypedef np.longlong_t LONGLONGTYPE_t  # WARNING: should match DTYPE in typedefs.pyx

# Fused type for certain operations
ctypedef fused DITYPE_t:
    ITYPE_t
    DTYPE_t
