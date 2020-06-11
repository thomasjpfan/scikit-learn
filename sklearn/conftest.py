import os
import sys
from distutils.version import LooseVersion
import platform

import pytest
from sklearn.externals import _pilutil
from sklearn.utils import _IS_32BIT
from _pytest.doctest import DoctestItem

from threadpoolctl import threadpool_limits

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


PYTEST_MIN_VERSION = '3.3.0'

if LooseVersion(pytest.__version__) < PYTEST_MIN_VERSION:
    raise ImportError('Your version of pytest is too old, you should have '
                      'at least pytest >= {} installed.'
                      .format(PYTEST_MIN_VERSION))


@pytest.fixture(scope='function')
def pyplot():
    """Setup and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we setup matplotlib backend and close the figures
    after running the functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('agg', warn=False, force=True)
    pyplot = pytest.importorskip('matplotlib.pyplot')
    yield pyplot
    pyplot.close('all')


def pytest_runtest_setup(item):
    """Set the number of openmp threads based on the number of workers
    xdist is using to prevent oversubscription.

    Parameters
    ----------
    item : pytest item
        item to be processed
    """
    try:
        xdist_worker_count = int(os.environ['PYTEST_XDIST_WORKER_COUNT'])
    except KeyError:
        # raises when pytest-xdist is not installed
        return

    openmp_threads = _openmp_effective_n_threads()
    threads_per_worker = max(openmp_threads // xdist_worker_count, 1)
    threadpool_limits(threads_per_worker, user_api='openmp')


def pytest_collection_modifyitems(config, items):

    # FeatureHasher is not compatible with PyPy
    if platform.python_implementation() == 'PyPy':
        skip_marker = pytest.mark.skip(
            reason='FeatureHasher is not compatible with PyPy')
        for item in items:
            if item.name.endswith(('_hash.FeatureHasher',
                                   'text.HashingVectorizer')):
                item.add_marker(skip_marker)

    # numpy changed the str/repr formatting of numpy arrays in 1.14. We want to
    # run doctests only for numpy >= 1.14.
    skip_doctests = False
    try:
        import numpy as np
        if LooseVersion(np.__version__) < LooseVersion('1.14'):
            reason = 'doctests are only run for numpy >= 1.14'
            skip_doctests = True
        elif _IS_32BIT:
            reason = ('doctest are only run when the default numpy int is '
                      '64 bits.')
            skip_doctests = True
        elif sys.platform.startswith("win32"):
            reason = ("doctests are not run for Windows because numpy arrays "
                      "repr is inconsistent across platforms.")
            skip_doctests = True
    except ImportError:
        pass

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)
    elif not _pilutil.pillow_installed:
        skip_marker = pytest.mark.skip(reason="pillow (or PIL) not installed!")
        for item in items:
            if item.name in [
                    "sklearn.feature_extraction.image.PatchExtractor",
                    "sklearn.feature_extraction.image.extract_patches_2d"]:
                item.add_marker(skip_marker)
