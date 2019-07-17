from .. import auc
from .. import roc_curve

from ...utils import check_matplotlib_support  # noqa


class RocCurveVisualizer:
    """ROC Curve visualization

    Parameters
    ----------

    estimator : estimator instance
        Trained classifier.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input values.

    y : array-like, shape (n_samples, )
        Target values.

    pos_label : int str or None, default=None
        The label of the positive class.
        When `pos_label=None`, if y_true is in {-1, 1} or {0, 1},
        `pos_label` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    drop_intermediate : boolean, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

    response_method : 'predict_proba', 'decision_function', or 'auto' \
    default='auto'
        Method to call estimator to get target scores

    Attributes
    ----------
    fpr_ : ndarray
        False positive rate.
    tpr_ : ndarray
        True positive rate.
    auc_ :
        Area under ROC curve.
    estimator_name_ : str
        Name of estimator.
    line_ : matplotlib Artist
        ROC Curve.
    ax_ : matplotlib Axes
        Axes with ROC curv
    figure_ : matplotlib Figure
        Figure containing the curve
    """

    def __init__(self, estimator, X, y, *, pos_label=None, sample_weight=None,
                 drop_intermediate=True, response_method="auto"):
        """Computes and stores values needed for visualization"""

        if response_method != "auto":
            prediction_method = getattr(estimator, response_method, None)
            if prediction_method is None:
                raise ValueError(
                    "response method {} not defined".format(response_method))
        else:
            predict_proba = getattr(estimator, 'predict_proba', None)
            decision_function = getattr(estimator, 'decision_function', None)
            prediction_method = predict_proba or decision_function

        if prediction_method is None:
            if response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method')
            else:
                raise ValueError(
                    'The estimator has no decision_function method')

        y_pred = prediction_method(X)

        if y_pred.ndim != 1:
            if y_pred.shape[1] > 2:
                raise ValueError("Estimator must be a binary classifier")
            y_pred = y_pred[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred, pos_label=pos_label,
                                drop_intermediate=drop_intermediate)

        self.fpr_ = fpr
        self.tpr_ = tpr
        self.auc_ = auc(fpr, tpr)
        self.estimator_name_ = estimator.__class__.__name__

    def plot(self, ax=None, line_kw=None):
        """Plot visualization

        Parameters
        ----------
        ax : Matplotlib Axes, optional (default=None)
            Axes object to plot on.

        line_kw : dict or None, optional (default=None)
            Keyword arguments to pass to
        """
        check_matplotlib_support('plot_roc_curve')  # noqa
        import matplotlib.pyplot as plt  # noqa

        if ax is None:
            fig, ax = plt.subplots()

        label = "{} (AUC = {:0.2f})".format(self.estimator_name_,
                                            self.auc_)
        line_kwargs = {"label": label}
        if line_kw is not None:
            line_kwargs.update(**line_kw)
        self.line_ = ax.plot(self.fpr_, self.tpr_, **line_kwargs)[0]
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


def plot_roc_curve(estimator, X, y, pos_label=None, sample_weight=None,
                   drop_intermediate=True, response_method="auto",
                   ax=None, line_kw=None):
    """Plot Receiver operating characteristic (ROC) curve

    Note: this implementation is restricted to the binary classification task.

    Read more in the :ref:`User Guide <roc_metrics>`.

    Parameters
    ----------

    estimator : estimator instance
        Trained classifier.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input values.

    y : array-like, shape (n_samples,, )
        Target values.

    pos_label : int or str, default=None
        The label of the positive class.
        When `pos_label=None`, if y_true is in {-1, 1} or {0, 1},
        `pos_label` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape = [n_samples], optional (default=None)
        Sample weights.

    drop_intermediate : boolean, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.

    response_method : 'predict_proba', 'decision_function', or 'auto' \
    default='auto'
        Method to call estimator to get target scores

    ax : matplotlib axes, default=None
        axes object to plot on

    line_kw : dict or None, default=None
        Keyword arguments to pass to

    Returns
    -------
    viz : :class:`sklearn.metrics.plot.RocCurveVisualizer`
        object that stores computed values
    """
    viz = RocCurveVisualizer(estimator,
                             X,
                             y,
                             sample_weight=sample_weight,
                             pos_label=pos_label,
                             drop_intermediate=drop_intermediate,
                             response_method=response_method)
    viz.plot(ax=ax, line_kw=line_kw)
    return viz
