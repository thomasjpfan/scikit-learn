import numpy as np

from ._encoders import _BaseEncoder
from ..base import OneToOneFeatureMixin
from ._target_encoder_fast import _fit_encoding_fast
from ..utils.validation import _check_y, check_consistent_length


class TargetRegressorEncoder(OneToOneFeatureMixin, _BaseEncoder):
    def __init__(self, categories="auto", smooth=30, cv=5):
        self.categories = categories
        self.smooth = smooth
        self.cv = cv

    def fit(self, X, y):
        self._fit_encodings_all(X, y)
        return self

    def _fit_encodings_all(self, X, y):
        # TODO: validate parameters
        check_consistent_length(X, y)
        self._fit(X, handle_unknown="ignore", force_all_finite="allow-nan")
        y = _check_y(y, y_numeric=True, estimator=self).astype(np.float64, copy=False)
        self.encoding_mean_ = np.mean(y)

        X_int, X_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        n_categories = np.asarray(
            [len(cat) for cat in self.categories_], dtype=np.int64
        )
        self.encodings_ = _fit_encoding_fast(
            X_int, y, n_categories, self.smooth, self.encoding_mean_
        )
        return X_int, X_mask, y, n_categories

    def fit_transform(self, X, y):
        from ..model_selection._split import check_cv  # avoid circular input

        X_int, X_mask, y, n_categories = self._fit_encodings_all(X, y)
        cv = check_cv(self.cv)
        X_out = np.empty_like(X_int, dtype=np.float64)
        X_invalid = ~X_mask

        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X_int[train_idx, :], y[train_idx]
            y_mean = np.mean(y_train)
            encodings = _fit_encoding_fast(
                X_train, y_train, n_categories, self.smooth, y_mean
            )
            self._transform_X_int(X_out, X_int, X_invalid, test_idx, encodings, y_mean)
        return X_out

    def transform(self, X, y=None):
        X_int, X_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        X_out = np.empty_like(X_int, dtype=np.float64)
        self._transform_X_int(
            X_out, X_int, ~X_mask, slice(None), self.encodings_, self.encoding_mean_
        )
        return X_out

    def _transform_X_int(self, X_out, X_int, X_invalid, indicies, encodings, y_mean):
        for f_idx, encoding in enumerate(encodings):
            X_out[indicies, f_idx] = np.take(encoding, X_int[indicies, f_idx])
            X_out[X_invalid[:, f_idx], f_idx] = y_mean
