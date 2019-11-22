"""Transformers for missing value imputation"""

import importlib
import sys

from ._base import MissingIndicator, SimpleImputer
from ._knn import KNNImputer
from ..externals._pep562 import Pep562

__all__ = [
    'MissingIndicator',
    'SimpleImputer',
    'KNNImputer'
]


# TODO: Remove when IterativeImputer is not experimental
# Enables better error messages when trying to import experimental features
def __getattr__(name):
    if name in __all__:
        return importlib.import_module("." + name, __name__)

    if name in ['IterativeImputer']:
        raise ImportError("To enable this experimental feature, run "
                          "'from sklearn.experimental import "
                          "enable_iterative_imputer' before "
                          "importing {name!r}".format(name=name))

    raise ImportError("cannot import name {name!r} from {__name__!r}".format(
        name=name, __name__=__name__))


if not sys.version_info >= (3, 7):
    Pep562(__name__)
