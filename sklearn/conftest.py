from contextlib import suppress
import platform
import pytest

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_kddcup99
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_rcv1


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


def pytest_addoption(parser):
    # Adding pytest again for test run with --pyargs
    with suppress(ValueError):
        parser.addoption("--run-network", action="store_true",
                        help="run network tests")


# fetching a dataset with this fixture will never download if missing
def _fetch_fixture(f):
    def wrapped(*args, **kwargs):
        kwargs['download_if_missing'] = False
        try:
            return f(*args, **kwargs)
        except IOError:
            pytest.skip("test requires --run-network to run")
    return pytest.fixture(lambda: wrapped)


dataset_fetchers = {
    'fetch_20newsgroups_fxt': fetch_20newsgroups,
    'fetch_20newsgroups_vectorized_fxt': fetch_20newsgroups_vectorized,
    'fetch_california_housing_fxt': fetch_california_housing,
    'fetch_covtype_fxt': fetch_covtype,
    'fetch_kddcup99_fxt': fetch_kddcup99,
    'fetch_olivetti_faces_fxt': fetch_olivetti_faces,
    'fetch_rcv1_fxt': fetch_rcv1,
}


fetch_20newsgroups_fxt = _fetch_fixture(fetch_20newsgroups)
fetch_20newsgroups_vectorized_fxt = \
    _fetch_fixture(fetch_20newsgroups_vectorized)
fetch_california_housing_fxt = _fetch_fixture(fetch_california_housing)
fetch_covtype_fxt = _fetch_fixture(fetch_covtype)
fetch_kddcup99_fxt = _fetch_fixture(fetch_kddcup99)
fetch_olivetti_faces_fxt = _fetch_fixture(fetch_olivetti_faces)
fetch_rcv1_fxt = _fetch_fixture(fetch_rcv1)


def pytest_collection_modifyitems(config, items):

    # FeatureHasher is not compatible with PyPy
    if platform.python_implementation() == 'PyPy':
        skip_marker = pytest.mark.skip(
            reason='FeatureHasher is not compatible with PyPy')
        for item in items:
            if item.name.endswith(('_hash.FeatureHasher',
                                   'text.HashingVectorizer')):
                item.add_marker(skip_marker)

    run_network_tests = config.getoption("--run-network")
    skip_network = pytest.mark.skip(
        reason="test requires --run-network to run")

    # download datasets during collection to avoid thread unsafe behavior
    # when running pytest in parallel with pytest-xdist
    dataset_features_set = set(dataset_fetchers)
    datasets_to_download = set()

    for item in items:
        dataset_to_fetch = set(item.keywords) & dataset_features_set
        if not dataset_to_fetch:
            continue

        if run_network_tests:
            datasets_to_download |= dataset_to_fetch
        else:
            # network tests are skipped
            item.add_marker(skip_network)

    # download datasets that are needed to avoid thread unsafe behavior
    # by pytest-xdist
    if run_network_tests:
        for name in datasets_to_download:
            dataset_fetchers[name]()
