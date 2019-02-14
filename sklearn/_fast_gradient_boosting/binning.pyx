# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
"""
This module contains the BinMapper class.

BinMapper is used for mapping a real-valued dataset into integer-valued bins
with equally-spaced thresholds.
"""
cimport cython

import numpy as np
cimport numpy as np
from cython.parallel import prange

from ..utils import check_random_state, check_array
from ..utils.validation import check_is_fitted
from ..base import BaseEstimator, TransformerMixin
from .types import X_DTYPE, X_BINNED_DTYPE
from .types cimport X_DTYPE_C, X_BINNED_DTYPE_C


def _find_binning_thresholds(data, max_bins=256, subsample=int(2e5),
                             random_state=None):
    """Extract feature-wise equally-spaced quantiles from numerical data


    Return
    ------
    binning_thresholds: tuple of arrays
        For each feature, stores the increasing numeric values that can
        be used to separate the bins. len(binning_thresholds) == n_features.
    """
    if not (2 <= max_bins <= 256):
        raise ValueError('max_bins={} should be no smaller than 2 '
                         'and no larger than 256.'.format(max_bins))
    rng = check_random_state(random_state)
    if subsample is not None and data.shape[0] > subsample:
        subset = rng.choice(np.arange(data.shape[0]), subsample)
        data = data[subset]

    percentiles = np.linspace(0, 100, num=max_bins + 1)
    end = percentiles.shape[0]  # no negative indexing!
    percentiles = percentiles[1:end - 1]
    binning_thresholds = []
    for f_idx in range(data.shape[1]):
        col_data = np.ascontiguousarray(data[:, f_idx], dtype=X_DTYPE)
        distinct_values = np.unique(col_data)
        if len(distinct_values) <= max_bins:
            end = distinct_values.shape[0]  # no negative indexing!
            midpoints = (distinct_values[:end - 1] + distinct_values[1:])
            midpoints *= .5
        else:
            # We sort again the data in this case. We could compute
            # approximate midpoint percentiles using the output of
            # np.unique(col_data, return_counts) instead but this is more
            # work and the performance benefit will be limited because we
            # work on a fixed-size subsample of the full data.
            midpoints = np.percentile(col_data, percentiles,
                                      interpolation='midpoint').astype(X_DTYPE)
        binning_thresholds.append(midpoints)
    return binning_thresholds


cpdef _map_to_bins(const X_DTYPE_C [:, :] data, list binning_thresholds,
                   X_BINNED_DTYPE_C [::1, :] binned):
    """Bin numerical values to discrete integer-coded levels.

    Parameters
    ----------
    data : array-like, shape=(n_samples, n_features)
        The numerical data to bin.
    binning_thresholds : tuple of arrays
        For each feature, stores the increasing numeric values that are
        used to separate the bins.
    out : array-like
        If not None, write result inplace in out.

    Returns
    -------
    binned_data : array of int, shape=data.shape
        The binned data.
    """
    cdef:
        int feature_idx

    for feature_idx in range(data.shape[1]):
        _map_num_col_to_bins(data[:, feature_idx],
                             binning_thresholds[feature_idx],
                             binned[:, feature_idx])


cpdef void _map_num_col_to_bins(const X_DTYPE_C [:] data,
                                const X_DTYPE_C [:] binning_thresholds,
                                X_BINNED_DTYPE_C [:] binned):
    """Binary search to the find the bin index for each value in data."""
    cdef:
        int i
        int left
        int right
        int middle

    for i in prange(data.shape[0], schedule='static', nogil=True):
        left, right = 0, binning_thresholds.shape[0]
        while left < right:
            middle = (right + left - 1) // 2
            if data[i] <= binning_thresholds[middle]:
                right = middle
            else:
                left = middle + 1
        binned[i] = left


class BinMapper(BaseEstimator, TransformerMixin):
    """Transformer that maps a dataset into integer-valued bins.

    The bins are created in a feature-wise fashion, with equally-spaced
    quantiles.

    Large datasets are subsampled, but the feature-wise quantiles should
    remain stable.

    If the number of unique values for a given feature is less than
    ``max_bins``, then the unique values of this feature are used instead of
    the quantiles.

    Parameters
    ----------
    max_bins : int, optional (default=256)
        The maximum number of bins to use. If for a given feature the number of
        unique values is less than ``max_bins``, then those unique values
        will be used to compute the bin thresholds, instead of the quantiles.
    subsample : int or None, optional (default=1e5)
        If ``n_samples > subsample``, then ``sub_samples`` samples will be
        randomly choosen to compute the quantiles. If ``None``, the whole data
        is used.
    random_state: int or numpy.random.RandomState or None, \
        optional (default=None)
        Pseudo-random number generator to control the random sub-sampling.
        See `scikit-learn glossary
        <https://scikit-learn.org/stable/glossary.html#term-random-state>`_.
    """
    def __init__(self, max_bins=256, subsample=int(1e5), random_state=None):
        self.max_bins = max_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit data X by computing the binning thresholds.

        Parameters
        ----------
        X: array-like
            The data to bin

        Returns
        -------
        self : object
        """
        X = check_array(X, dtype=[X_DTYPE])
        self.bin_thresholds_ = _find_binning_thresholds(
            X, self.max_bins, subsample=self.subsample,
            random_state=self.random_state)

        self.n_bins_per_feature_ = np.array(
            [thresholds.shape[0] + 1 for thresholds in self.bin_thresholds_],
            dtype=np.uint32)

        return self

    def transform(self, X):
        """Bin data X.

        Parameters
        ----------
        X: array-like
            The data to bin

        Returns
        -------
        X_binned : array-like
            The binned data
        """
        X = check_array(X, dtype=[X_DTYPE])
        check_is_fitted(self, ['bin_thresholds_', 'n_bins_per_feature_'])
        if X.shape[1] != self.n_bins_per_feature_.shape[0]:
            raise ValueError(
                'This estimator was fitted with {} features but {} got passed '
                'to transform()'.format(self.n_bins_per_feature_.shape[0],
                                        X.shape[1])
            )
        binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order='F')
        _map_to_bins(X, self.bin_thresholds_, binned)
        return binned
