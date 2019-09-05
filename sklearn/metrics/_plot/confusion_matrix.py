from itertools import product

import numpy as np

from ..metrics import confusion_matrix
from ...utils import check_matplotlib_support
from ...utils.multiclass import unique_labels
from ...base import is_classifier


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    target_names : ndarray of shape (n_classes,)
        Target names.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, target_names):
        self.confusion_matrix = confusion_matrix
        self.target_names = target_names

    def plot(self, include_values=True,
             normalize=False, cmap='viridis', ax=None):
        """Plot visualization.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, None]

        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        values = np.empty(cm.shape, dtype=np.object)
        if include_values:
            fmt = '.2f' if normalize else 'd'
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):





        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.target_names,
               yticklabels=self.target_names,
               ylabel="True label",
               xlabel="Predicted label")

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(estimator, X, y_true, sample_weight=None,
                          labels=None, target_names=None,
                          include_values=True, normalize=False,
                          cmap='viridis', ax=None):
    """Plot Confusion Matrix.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    estimator : estimator instance
        Trained classifier.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.

    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.

    target_names : array-like of shape (n_classes,), default=None
        Target names used for plotting.

    normalize : bool, default=False
        Normalizes confusion matrix.

    include_values : bool, default=True
        Includes values in confusion matrix.

    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.

    ax : matplotlib Axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created.

    Returns
    -------
    display: :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    """
    check_matplotlib_support("plot_confusion_matrix")

    if not is_classifier(estimator):
        raise ValueError("plot_confusion_matrix only supports classifiers")

    y_pred = estimator.predict(X)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels)

    if target_names is None:
        if labels is None:
            target_names = labels
        else:
            target_names = unique_labels(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  target_names=target_names)
    return disp.plot(include_values=include_values, normalize=normalize,
                     cmap=cmap, ax=ax)
