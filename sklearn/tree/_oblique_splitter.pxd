# distutils: language = c++

# Authors: Adam Li <adam2392@gmail.com>
#          Chester Huynh <chester.huynh924@gmail.com>
#          Parth Vora <pvora4@jhu.edu>
#
# License: BSD 3 clause

# See _oblique_splitter.pyx for details.

import numpy as np
cimport numpy as np

from ._criterion cimport Criterion
from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

from libcpp.vector cimport vector

cdef struct ObliqueSplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.

    vector[DTYPE_t]* proj_vec_weights   # weights of the vector
    vector[SIZE_t]* proj_vec_indices    # indices of the features


cdef class BaseObliqueSplitter(Splitter):
    cdef public double feature_combinations
    cdef vector[vector[DTYPE_t]] proj_mat_weights       # nonzero weights of sparse proj_mat matrix
    cdef vector[vector[SIZE_t]] proj_mat_indices        # nonzero indices of sparse proj_mat matrix
    cdef SIZE_t n_non_zeros

    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1

    # All oblique splitters (i.e. non-axis aligned splitters) require a
    # function to sample a projection matrix that is applied to the feature matrix
    # to quickly obtain the sampled projections for candidate splits.
    cdef void sample_proj_mat(self,
                              vector[vector[DTYPE_t]]& proj_mat_weights,
                              vector[vector[SIZE_t]]& proj_mat_indices) nogil
