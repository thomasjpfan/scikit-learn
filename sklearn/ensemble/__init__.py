"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""

import sys
import importlib

from ._base import BaseEnsemble
from ._forest import RandomForestClassifier
from ._forest import RandomForestRegressor
from ._forest import RandomTreesEmbedding
from ._forest import ExtraTreesClassifier
from ._forest import ExtraTreesRegressor
from ._bagging import BaggingClassifier
from ._bagging import BaggingRegressor
from ._iforest import IsolationForest
from ._weight_boosting import AdaBoostClassifier
from ._weight_boosting import AdaBoostRegressor
from ._gb import GradientBoostingClassifier
from ._gb import GradientBoostingRegressor
from ._voting import VotingClassifier
from ._voting import VotingRegressor
from ._stacking import StackingClassifier
from ._stacking import StackingRegressor
from ..externals._pep562 import Pep562

from . import partial_dependence

__all__ = ["BaseEnsemble",
           "RandomForestClassifier", "RandomForestRegressor",
           "RandomTreesEmbedding", "ExtraTreesClassifier",
           "ExtraTreesRegressor", "BaggingClassifier",
           "BaggingRegressor", "IsolationForest", "GradientBoostingClassifier",
           "GradientBoostingRegressor", "AdaBoostClassifier",
           "AdaBoostRegressor", "VotingClassifier", "VotingRegressor",
           "StackingClassifier", "StackingRegressor",
           "partial_dependence"]


# TODO: Remove when HistGradientBoosting* is not experimental
# Enables better error messages when trying to import experimental features
def __getattr__(name):
    if (name in ['HistGradientBoostingClassifier',
                 'HistGradientBoostingRegressor'] and
            name not in __all__):
        raise ImportError("To enable this experimental feature, run "
                          "'from sklearn.experimental import "
                          "enable_hist_gradient_boosting' before "
                          "importing {name!r}".format(name=name))
    return importlib.import_module(__name__, name)


if not sys.version_info >= (3, 7):
    Pep562(__name__)
