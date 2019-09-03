import pytest
import numpy as np
from numpy.testing import assert_allclose

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusions_matrix


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=100, n_informative=5, n_classes=5,
                               random_state=0)
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(data):
    return SVC(kernel='linear', C=0.01).fit(*data)


def test_error_on_regressor():
    pass


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("with_sample_weight", [True, False])
def test_plot_confusion_matrix(pyplot, data, fitted_clf, normalize,
                               with_sample_weight):
    X, y = data
    y_pred = fitted_clf.predict(X)

    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=X.shape[0])
    else:
        sample_weight = None

    cm = confusion_matrix(y, y_pred, sample_weight=sample_weight)

    viz = plot_confusions_matrix(fitted_clf, X, y,
                                 sample_weight=sample_weight,
                                 normalize=normalize)

    assert_allclose(viz.cm_, cm)


def test_with_labels():
    pass


def test_include_values(data, fitted_clf):
    pass


def test_include_colorbar():
    pass
