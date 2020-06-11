# Even if empty this file is useful so that when running from the root folder
# ./sklearn is added to sys.path by pytest. See
# https://docs.pytest.org/en/latest/pythonpath.html for more details.  For
# example, this allows to build extensions in place and run pytest
# doc/modules/clustering.rst and use sklearn from the local folder rather than
# the one from site-packages.

from distutils.version import LooseVersion

import pytest


PYTEST_MIN_VERSION = '3.3.0'

if LooseVersion(pytest.__version__) < PYTEST_MIN_VERSION:
    raise ImportError('Your version of pytest is too old, you should have '
                      'at least pytest >= {} installed.'
                      .format(PYTEST_MIN_VERSION))


def pytest_addoption(parser):
    parser.addoption("--skip-network", action="store_true", default=False,
                     help="skip network tests")


def pytest_collection_modifyitems(config, items):
    # Skip tests which require internet if the flag is provided
    if config.getoption("--skip-network"):
        skip_network = pytest.mark.skip(
            reason="test requires internet connectivity")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


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
