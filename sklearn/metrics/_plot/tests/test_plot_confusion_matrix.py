import pytest
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


@pytest.fixture(scope="module")
def n_classes()
    return 5

@pytest.fixture(scope="module")
def data(n_classes):
    X, y = make_classification(n_samples=100, n_informative=5,
                               n_classes=n_classes, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(data):
    return SVC(kernel='linear', C=0.01).fit(*data)


@pytest.fixture(scope="module")
def y_pred(data, fitted_clf):
    X, _ = data
    return fitted_clf.predict(X)


def test_error_on_regressor(pyplot, data):
    X, y = data
    est = SVR().fit(X, y)

    msg = "plot_confusion_matrix only supports classifiers"
    with pytest.raises(ValueError, match=msg):
        plot_confusion_matrix(est, X, y)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("with_sample_weight", [True, False])
@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("cmap", ['viridis', 'plasma'])
@pytest.mark.parametrize("with_custom_axes", [True, False])
@pytest.mark.parametrize("with_target_names", [True, False])
@pytest.mark.parametrize("include_values", [True, False])
def test_plot_confusion_matrix(pyplot, data, y_pred, n_classes, fitted_clf,
                               normalize, with_sample_weight, with_labels,
                               cmap, with_custom_axes, with_target_names,
                               include_values):
    X, y = data

    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=X.shape[0])
    else:
        sample_weight = None

    ax = pyplot.gca() if with_custom_axes else None

    labels = None if with_labels else [2, 1, 0, 3, 4]
    target_names = ['b', 'd', 'a', 'e', 'f'] if with_target_names else None

    cm = confusion_matrix(y, y_pred, sample_weight=sample_weight,
                          labels=labels)

    viz = plot_confusion_matrix(fitted_clf, X, y,
                                sample_weight=sample_weight,
                                normalize=normalize, labels=labels,
                                cmap=cmap, ax=ax, target_names=target_names,
                                include_values=include_values)

    if with_custom_axes:
        assert viz.ax_ == ax

    assert_allclose(viz.confusion_matrix_, cm)
    import matplotlib as mpl
    assert isinstance(viz.im_, mpl.image.AxesImage)
    assert viz.im_.get_cmap().name == cmap
    assert isinstance(viz.ax_, pyplot.Axes)
    assert isinstance(viz.figure_, pyplot.Figure)

    assert viz.im_.get_ylabel() == "True label"
    assert viz.im_.get_xlabel() == "Predicted label"

    x_ticks = [tick.get_text() for tick in viz.ax_.get_xtickslabels()]
    y_ticks = [tick.get_text() for tick in viz.ax_.get_ytickslabels()]

    if with_target_names:
        expected_target_names = target_names
    elif with_labels:
        expected_target_names = [str(i) for i in labels]
    else:
        expected_target_names = [str(i) for i in range(n_classes)]

    assert x_ticks == expected_target_names
    assert y_ticks == expected_target_names

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, None]

    image_data = viz.im_.get_array().data
    assert_allclose(image_data, cm)

    if include_values:
        assert viz.values_.shape == (n_classes, n_classes)
        fmt = '.2f' if normalize else 'd'
        expected_text = np.array([format(v, fmt) for v in cm.ravel(order="C")])
        values_text = np.array([
            t.get_text() for t in viz.values_.ravel(order="C")])
        assert_array_equal(expected_text, values_text)
    else:
        assert viz.values_ is None


def test_confusion_matrix_display(pyplot, data, fitted_clf, y_pred, n_classes):
    X, y = data

    cm = confusion_matrix(y, y_pred)
    viz = plot_confusion_matrix(fitted_clf, X, y, normalize=False,
                                include_values=True, cmap='viridis')

    assert_allclose(viz.confusion_matrix_, cm)
    assert viz.values_ == (n_classes, n_classes)

    image_data = viz.im_.get_array().data
    assert_allclose(image_data, cm)

    viz.plot(cmap='plasma')
    assert viz.im_.get_cmap().name == 'plasma'

    viz.plot(include_values=False)
    assert viz.values_ is None

    viz.plot(normalize=True)
    image_data = viz.im_.get_array().data
    assert_allclose(image_data, cm.astype('float') / cm.sum(axis=1)[:, None])
