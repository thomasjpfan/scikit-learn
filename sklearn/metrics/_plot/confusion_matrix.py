class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.

    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a visualizer. All parameters are stored as attributes.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    target_names : array-like of shape (n_classes,)
        Target names.

    Attributes
    ----------
    ax_ : matplotlib Axes
        Axes with confusion matrix.

    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, target_names):
        pass

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
        pass


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

    normalize : bool, default=False
        Normalizes confusion matrix.

    include_values : bool, default=True
        Includes values in confusion matrix.

    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.

    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created.

    Returns
    -------
    display: :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    """
    pass
