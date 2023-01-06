from ._encoders import _BaseEncoder
from ..base import OneToOneFeatureMixin

# from ..model_selection import check_cv
# from ..utils.validation import _check_y


class TargetRegressorEncoder(OneToOneFeatureMixin, _BaseEncoder):
    def __init__(self, categories="auto", smooth=30, cv=5):
        self.categories = categories
        self.smooth = smooth
        self.cv = cv

    def fit(self, X, y):
        return self

    def fit_tranfsorm(self, X, y):
        return X

    def transform(self, X, y=None):
        return X
