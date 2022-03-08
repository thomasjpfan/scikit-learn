from cython.operator cimport dereference as deref
from cpython.ref cimport Py_INCREF
cimport numpy as np

from ._typedefs cimport DTYPECODE, FTYPECODE, INT64TYPECODE, ITYPECODE, INT32TYPECODE

np.import_array()

cdef int _get_type_num(vector_typed * vect_ptr):
    if vector_typed is vector[DTYPE_t]:
        return DTYPECODE
    elif vector_typed is vector[FTYPE_t]:
        return FTYPECODE
    elif vector_typed is vector[INT64_t]:
        return INT64TYPECODE
    elif vector_typed is vector[INT32_t]:
        return INT32TYPECODE
    else:
        return ITYPECODE

cdef StdVectorSentinel _create_sentinel(vector_typed * vect_ptr):
    if vector_typed is vector[DTYPE_t]:
        return StdVectorSentinelFloat64.create_for(vect_ptr)
    elif vector_typed is vector[FTYPE_t]:
        return StdVectorSentinelFloat32.create_for(vect_ptr)
    elif vector_typed is vector[INT64_t]:
        return StdVectorSentinelInt64.create_for(vect_ptr)
    elif vector_typed is vector[INT32_t]:
        return StdVectorSentinelInt32.create_for(vect_ptr)
    else:
        return StdVectorSentinelFloatIntP.create_for(vect_ptr)

cdef class StdVectorSentinel:
    """Wraps a reference to a vector which will be deallocated with this object.

    When created, the StdVectorSentinel swaps the reference of its internal
    vectors with the provided one (vec_ptr), thus making the StdVectorSentinel
    manage the provided one's lifetime.
    """
    pass


cdef class StdVectorSentinelFloat64(StdVectorSentinel):
    cdef vector[DTYPE_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[DTYPE_t] * vec_ptr):
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelFloat64 sentinel = StdVectorSentinelFloat64.__new__(StdVectorSentinelFloat64)
        sentinel.vec.swap(deref(vec_ptr))
        return sentinel

cdef class StdVectorSentinelFloat32(StdVectorSentinel):
    cdef vector[FTYPE_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[FTYPE_t] * vec_ptr):
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelFloat32 sentinel = StdVectorSentinelFloat32.__new__(StdVectorSentinelFloat32)
        sentinel.vec.swap(deref(vec_ptr))
        return sentinel

cdef class StdVectorSentinelInt64(StdVectorSentinel):
    cdef vector[INT64_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[INT64_t] * vec_ptr):
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelInt64 sentinel = StdVectorSentinelInt64.__new__(StdVectorSentinelInt64)
        sentinel.vec.swap(deref(vec_ptr))
        return sentinel

cdef class StdVectorSentinelFloatIntP(StdVectorSentinel):
    cdef vector[ITYPE_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[ITYPE_t] * vec_ptr):
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelFloatIntP sentinel = StdVectorSentinelFloatIntP.__new__(StdVectorSentinelFloatIntP)
        sentinel.vec.swap(deref(vec_ptr))
        return sentinel


cdef class StdVectorSentinelInt32(StdVectorSentinel):
    cdef vector[INT32_t] vec

    @staticmethod
    cdef StdVectorSentinel create_for(vector[INT32_t] * vec_ptr):
        # This initializes the object directly without calling __init__
        # See: https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#instantiation-from-existing-c-c-pointers # noqa
        cdef StdVectorSentinelInt32 sentinel = StdVectorSentinelInt32.__new__(StdVectorSentinelInt32)
        sentinel.vec.swap(deref(vec_ptr))
        return sentinel


cdef np.ndarray vector_to_nd_array(vector_typed * vect_ptr):
    cdef:
        np.npy_intp size
        np.ndarray arr
        int typenum = _get_type_num(vect_ptr)
        StdVectorSentinel sentinel

    size = deref(vect_ptr).size()
    arr = np.PyArray_SimpleNewFromData(1, &size, typenum, deref(vect_ptr).data())

    sentinel = _create_sentinel(vect_ptr)

    # Makes the numpy array responsible of the life-cycle of its buffer.
    # A reference to the StdVectorSentinel will be stolen by the call to
    # `PyArray_SetBaseObject` below, so we increase its reference counter.
    # See: https://docs.python.org/3/c-api/intro.html#reference-count-details
    Py_INCREF(sentinel)
    np.PyArray_SetBaseObject(arr, sentinel)
    return arr
