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
    def __init__(self, confusion_matrix, classes):
        pass

    def plot(self, cmap='viridis', ax=None):
        """Plot visualization.

        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.

        include_colorbar : bool, default=True
            Includes colorbar next to confusion matrix.

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


def plot_confusions_matrix(estimator, y_true, sample_weight=None,
                           target_names=None,
                           include_values=True, normalize=False,
                           cmap='viridis', ax=None):
    """Plot Confusion Matrix.

    Read more in the :ref:`User Guide <visualizations>`.

    Parameters
    ----------
    normalize : bool, default=False
        Normalizes confusion matrix.

    include_values : bool, default=True
        Includes values in confusion matrix.

    include_colorbar : bool, default=True
        Includes colorbar next to confusion matrix.

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
