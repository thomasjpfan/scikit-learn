# Even if empty this file is useful so that when running from the root folder
# ./sklearn is added to sys.path by pytest. See
# https://docs.pytest.org/en/latest/pythonpath.html for more details.  For
# example, this allows to build extensions in place and run pytest
# doc/modules/clustering.rst and use sklearn from the local folder rather than
# the one from site-packages.

import platform
import sys
from distutils.version import LooseVersion
import os

import pytest
from _pytest.doctest import DoctestItem

from sklearn.utils import _IS_32BIT
from sklearn.externals import _pilutil
from threadpoolctl import threadpool_limits
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


PYTEST_MIN_VERSION = '3.3.0'

if LooseVersion(pytest.__version__) < PYTEST_MIN_VERSION:
    raise ImportError('Your version of pytest is too old, you should have '
                      'at least pytest >= {} installed.'
                      .format(PYTEST_MIN_VERSION))


def pytest_addoption(parser):
    parser.addoption("--skip-network", action="store_true", default=False,
                     help="skip network tests")


def pytest_collection_modifyitems(config, items):

    # FeatureHasher is not compatible with PyPy
    if platform.python_implementation() == 'PyPy':
        skip_marker = pytest.mark.skip(
            reason='FeatureHasher is not compatible with PyPy')
        for item in items:
            if item.name.endswith(('_hash.FeatureHasher',
                                   'text.HashingVectorizer')):
                item.add_marker(skip_marker)

    # Skip tests which require internet if the flag is provided
    if config.getoption("--skip-network"):
        skip_network = pytest.mark.skip(
            reason="test requires internet connectivity")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)

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


@pytest.fixture(autouse=True, scope='session')
def set_openmp_threadpool():
    """Set the number of openmp threads based on the number of workers
    xdist is using to prevent oversubscription."""
    xdist_worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 1))
    openmp_threads = _openmp_effective_n_threads()
    threads_per_worker = max(openmp_threads // xdist_worker_count, 1)
    threadpool_limits(threads_per_worker, user_api='openmp')


def pytest_configure(config):
    import sys
    sys._is_pytest_session = True
    # declare our custom markers to avoid PytestUnknownMarkWarning
    config.addinivalue_line(
        "markers",
        "network: mark a test for execution if network available."
    )


def pytest_unconfigure(config):
    import sys
    del sys._is_pytest_session
