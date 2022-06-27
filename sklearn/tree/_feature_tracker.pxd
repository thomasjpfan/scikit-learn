# Feature Tracker used with the splitter to sample features
cimport cython
from ._tree cimport SIZE_t
from ._tree cimport UINT32_t

cdef enum FeatureStatus:
    EVALUTE, STOP, CONTINUE

cdef struct FeatureSample:
    SIZE_t f_j
    SIZE_t feature
    FeatureStatus status

cdef struct FeatureTracker:
    SIZE_t* features
    SIZE_t* constant_features
    SIZE_t f_i
    SIZE_t max_features
    SIZE_t n_visited_features
    # Number of features discovered to be constant during the split search
    SIZE_t n_found_constants
    # Number of features known to be constant and drawn without replacement
    SIZE_t n_drawn_constants
    SIZE_t n_known_constants
    # n_total_constants = n_known_constants + n_found_constants
    SIZE_t n_total_constants

cdef void init_tracker(FeatureTracker* self, SIZE_t[::1] features,
                       SIZE_t[::1] constant_features, SIZE_t max_features,
                       SIZE_t n_constant_features) nogil
cdef FeatureSample sample_feature(FeatureTracker* self, UINT32_t* random_state) nogil
cdef void update_found_constant(FeatureTracker* self, SIZE_t f_j) nogil
cdef void update_drawn_feature(FeatureTracker* self, SIZE_t f_j) nogil
cdef SIZE_t update_constant_features(FeatureTracker* self) nogil
