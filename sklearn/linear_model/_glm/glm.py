"""
Generalized Linear Models with Exponential Dispersion Family
"""

# Author: Christian Lorentzen <lorentzen.ch@gmail.com>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause

import numbers

import numpy as np
import scipy.optimize

from ..._loss.loss import (
    BaseLoss,
    HalfGammaLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
)
from ...base import BaseEstimator, RegressorMixin
from ...utils.optimize import _check_optimize_result
from ...utils import check_scalar, check_array, deprecated
from ...utils.validation import check_is_fitted, _check_sample_weight
from ...utils._openmp_helpers import _openmp_effective_n_threads
from .._linear_loss import LinearModelLoss


class HalfInverseGaussianLoss(HalfTweedieLoss):
    """Half inverse Gaussian deviance loss with log-link, for regression.

    This is equivalent to HalfTweedieLoss(power=3).
    """

    def __init__(self, sample_weight=None):
        super().__init__(sample_weight=sample_weight, power=3)


# TODO: We could allow strings for _base_loss (as before for the now removed
# family parameter): 'normal', 'poisson', 'gamma', 'inverse-gaussian'
class _GeneralizedLinearRegressor(RegressorMixin, BaseEstimator):
    """Regression via a penalized Generalized Linear Model (GLM).

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at fitting and
    predicting the mean of the target y as y_pred=h(X*w) with coefficients w.
    Therefore, the fit minimizes the following objective function with L2 priors as
    regularizer::

        1/(2*sum(s_i)) * sum(s_i * deviance(y_i, h(x_i*w)) + 1/2 * alpha * ||w||_2^2

    with inverse link function h, s=sample_weight and per observation (unit) deviance
    deviance(y_i, h(x_i*w)). Note that for an EDM 1/2 * deviance is the negative
    log-likelihood up to a constant (in w) term.
    The parameter ``alpha`` corresponds to the lambda parameter in glmnet.

    Instead of implementing the EDM family and a link function seperately, we directly
    use the loss functions `from sklearn._loss` which have the link functions included
    in them for performance reasons. We pick the loss functions that implement
    (1/2 times) EDM deviances.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    family : {'normal', 'poisson', 'gamma', 'inverse-gaussian'} \
            or a BaseLoss instance, default='normal'
        The distributional assumption of the GLM, i.e. which distribution from
        the EDM, specifies the loss function to be minimized.

        .. deprecated:: 1.1
           `family` is deprecated in 1.1 and will be removed in 1.3.
           Use `_base_loss` instead.

    _base_loss : BaseLoss, default=HalfSquaredError()
        A `_base_loss` contains a specific loss function as well as the link
        function. The loss to be minimized specifies the distributional assumption of
        the GLM, i.e. the distribution from the EDM. Here are some examples:

        =======================  ========  ==========================
        _base_loss               Link      Target Domain
        =======================  ========  ==========================
        HalfSquaredError         identity  y any real number
        HalfPoissonLoss          log       0 <= y
        HalfGammaLoss            log       0 < y
        HalfInverseGaussianLoss  log       0 < y
        HalfTweedieLoss          log       dependend on tweedie power
        HalfTweedieLossIdentity  identity  dependend on tweedie power
        =======================  ========  ==========================

        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. For instance, with a log link,
        we have `y_pred = exp(X @ coeff + intercept)`.

    solver : 'lbfgs', default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_``.

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        _base_loss=HalfSquaredError(),
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._base_loss = _base_loss
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a Generalized Linear Model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted model.
        """
        check_scalar(
            self.alpha,
            name="alpha",
            target_type=numbers.Real,
            min_val=0.0,
            include_boundaries="left",
        )
        if not isinstance(self.fit_intercept, bool):
            raise ValueError(
                "The argument fit_intercept must be bool; got {0}".format(
                    self.fit_intercept
                )
            )
        if self.solver not in ["lbfgs"]:
            raise ValueError(
                f"{self.__class__.__name__} supports only solvers 'lbfgs'; "
                f"got {self.solver}"
            )
        solver = self.solver
        check_scalar(
            self.max_iter,
            name="max_iter",
            target_type=numbers.Integral,
            min_val=1,
        )
        check_scalar(
            self.tol,
            name="tol",
            target_type=numbers.Real,
            min_val=0.0,
            include_boundaries="neither",
        )
        check_scalar(
            self.verbose,
            name="verbose",
            target_type=numbers.Integral,
            min_val=0,
        )
        if not isinstance(self.warm_start, bool):
            raise ValueError(
                "The argument warm_start must be bool; got {0}".format(self.warm_start)
            )

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csc", "csr"],
            dtype=[np.float64, np.float32],
            y_numeric=True,
            multi_output=False,
        )

        # required by losses
        if solver == "lbfgs":
            # lbfgs will force coef and therefore raw_prediction to be float64. The
            # base_loss needs y, X @ coef and sample_weight all of same dtype
            # (and contiguous).
            loss_dtype = np.float64
        else:
            loss_dtype = min(max(y.dtype, X.dtype), np.float64)
        y = check_array(y, dtype=loss_dtype, order="C", ensure_2d=False)

        # TODO: We could support samples_weight=None as the losses support it.
        # Note that _check_sample_weight calls check_array(order="C") required by
        # losses.
        sample_weight = _check_sample_weight(sample_weight, X, dtype=loss_dtype)

        n_samples, n_features = X.shape

        self._linear_loss = LinearModelLoss(
            base_loss=self._validate_base_loss(),
            fit_intercept=self.fit_intercept,
        )

        if not self._linear_loss.base_loss.in_y_true_range(y):
            raise ValueError(
                "Some value(s) of y are out of the valid range of the loss"
                f" {self._base_loss.__class__.__name__!r}."
            )

        # TODO: if alpha=0 check that X is not rank deficient

        # IMPORTANT NOTE: Rescaling of sample_weight:
        # We want to minimize
        #     obj = 1/(2*sum(sample_weight)) * sum(sample_weight * deviance)
        #         + 1/2 * alpha * L2,
        # with
        #     deviance = 2 * loss.
        # The objective is invariant to multiplying sample_weight by a constant. We
        # choose this constant such that sum(sample_weight) = 1. Thus, we end up with
        #     obj = sum(sample_weight * loss) + 1/2 * alpha * L2.
        # Note that LinearModelLoss.loss() computes sum(sample_weight * loss).
        sample_weight = sample_weight / sample_weight.sum()

        if self.warm_start and hasattr(self, "coef_"):
            if self.fit_intercept:
                # LinearModelLoss needs intercept at the end of coefficient array.
                coef = np.concatenate((self.coef_, np.array([self.intercept_])))
            else:
                coef = self.coef_
            coef = coef.astype(loss_dtype, copy=False)
        else:
            if self.fit_intercept:
                coef = np.zeros(n_features + 1, dtype=loss_dtype)
                coef[-1] = self._linear_loss.base_loss.link.link(
                    np.average(y, weights=sample_weight)
                )
            else:
                coef = np.zeros(n_features, dtype=loss_dtype)

        # Algorithms for optimization:
        # Note again that our losses implement 1/2 * deviance.
        if solver == "lbfgs":
            func = self._linear_loss.loss_gradient
            l2_reg_strength = self.alpha
            n_threads = _openmp_effective_n_threads()

            opt_res = scipy.optimize.minimize(
                func,
                coef,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": self.max_iter,
                    "iprint": (self.verbose > 0) - 1,
                    "gtol": self.tol,
                    "ftol": 1e3 * np.finfo(float).eps,
                },
                args=(X, y, sample_weight, l2_reg_strength, n_threads),
            )
            self.n_iter_ = _check_optimize_result("lbfgs", opt_res)
            coef = opt_res.x

        if self.fit_intercept:
            self.intercept_ = coef[-1]
            self.coef_ = coef[:-1]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.0
            self.coef_ = coef

        return self

    def _linear_predictor(self, X):
        """Compute the linear_predictor = `X @ coef_ + intercept_`.

        Note that we often use the term raw_prediction instead of linear predictor.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values of linear predictor.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            allow_nd=False,
            reset=False,
        )
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predict using GLM with feature matrix X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values.
        """
        # check_array is done in _linear_predictor
        raw_prediction = self._linear_predictor(X)
        y_pred = self._linear_loss.base_loss.link.inverse(raw_prediction)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Compute D^2, the percentage of deviance explained.

        D^2 is a generalization of the coefficient of determination R^2.
        R^2 uses squared error and D^2 deviance. Note that those two are equal
        for ``_base_loss=HalfSquaredError()``.

        D^2 is defined as
        :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
        :math:`D_{null}` is the null deviance, i.e. the deviance of a model
        with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
        The mean :math:`\\bar{y}` is averaged by sample_weight.
        Best possible score is 1.0 and it can be negative (because the model
        can be arbitrarily worse).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values of target.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            D^2 of self.predict(X) w.r.t. y.
        """
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        raw_prediction = self._linear_predictor(X)  # validates X
        # required by losses
        y = check_array(y, dtype=raw_prediction.dtype, order="C", ensure_2d=False)

        if sample_weight is not None:
            # Note that _check_sample_weight calls check_array(order="C") required by
            # losses.
            sample_weight = _check_sample_weight(sample_weight, X, dtype=y.dtype)

        base_loss = self._linear_loss.base_loss

        if not base_loss.in_y_true_range(y):
            raise ValueError(
                "Some value(s) of y are out of the valid range of the loss"
                f" {self._base_loss.__name__}."
            )

        # Note that constant_to_optimal_zero is already multiplied by sample_weight.
        constant = np.mean(base_loss.constant_to_optimal_zero(y_true=y))
        if sample_weight is not None:
            constant *= sample_weight.shape[0] / np.sum(sample_weight)

        # Missing factor of 2 in deviance cancels out.
        deviance = base_loss(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=1,
        )
        y_mean = base_loss.link.link(np.average(y, weights=sample_weight))
        deviance_null = base_loss(
            y_true=y,
            raw_prediction=np.tile(y_mean, y.shape[0]),
            sample_weight=sample_weight,
            n_threads=1,
        )
        return 1 - (deviance + constant) / (deviance_null + constant)

    def _more_tags(self):
        # Create instance of BaseLoss if fit wasn't called yet. This is necessary as
        # TweedieRegressor might set the used loss during fit different from
        # self._base_loss.
        base_loss = self._validate_base_loss()
        return {"requires_positive_y": not base_loss.in_y_true_range(-1.0)}

    def _validate_base_loss(self):
        # This is only necessary because of the link and power arguments of the
        # TweedieRegressor.
        # Note that we do not need to pass sample_weight to the loss class as this is
        # only needed to set loss.constant_hessian on which GLMs do not rely.
        if not isinstance(self._base_loss, BaseLoss):
            raise ValueError(
                "The _base_loss must be an instance of sklearn._loss.loss.BaseLoss; "
                f"got _base_loss_class={self._base_loss}"
            )
        return self._base_loss

    # FIXME: remove in v1.3
    @deprecated(  # type: ignore
        "Attribute `family` was deprecated in version 1.1 and "
        "will be removed in 1.3. Use `_base_loss` instead."
    )
    @property
    def family(self):
        if isinstance(self, PoissonRegressor):
            return "poisson"
        elif isinstance(self, GammaRegressor):
            return "gamma"
        elif isinstance(self, TweedieRegressor) and self.power == 3:
            return "inverse-gaussian"
        else:
            return self._base_loss.__class__.__name__


class PoissonRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Poisson distribution.

    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Actual number of iterations used in the solver.

    See Also
    --------
    TweedieRegressor : Generalized Linear Model with a Tweedie distribution.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.PoissonRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [12, 17, 22, 21]
    >>> clf.fit(X, y)
    PoissonRegressor()
    >>> clf.score(X, y)
    0.990...
    >>> clf.coef_
    array([0.121..., 0.158...])
    >>> clf.intercept_
    2.088...
    >>> clf.predict([[1, 1], [3, 4]])
    array([10.676..., 21.875...])
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            _base_loss=HalfPoissonLoss(),
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _validate_base_loss(self):
        if not isinstance(self._base_loss, HalfPoissonLoss):
            raise ValueError("PoissonRegressor._base_loss must be HalfPoissonLoss!")
        return HalfPoissonLoss()


class GammaRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Gamma distribution.

    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X * coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_iter_ : int
        Actual number of iterations used in the solver.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PoissonRegressor : Generalized Linear Model with a Poisson distribution.
    TweedieRegressor : Generalized Linear Model with a Tweedie distribution.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.GammaRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [19, 26, 33, 30]
    >>> clf.fit(X, y)
    GammaRegressor()
    >>> clf.score(X, y)
    0.773...
    >>> clf.coef_
    array([0.072..., 0.066...])
    >>> clf.intercept_
    2.896...
    >>> clf.predict([[1, 0], [2, 8]])
    array([19.483..., 35.795...])
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            _base_loss=HalfGammaLoss(),
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _validate_base_loss(self):
        if not isinstance(self._base_loss, HalfGammaLoss):
            raise ValueError("GammaRegressor._base_loss must be HalfGammaLoss!")
        return HalfGammaLoss()


class TweedieRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Tweedie distribution.

    This estimator can be used to model different GLMs depending on the
    ``power`` parameter, which determines the underlying distribution.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    power : float, default=0
            The power determines the underlying target distribution according
            to the following table:

            +-------+------------------------+
            | Power | Distribution           |
            +=======+========================+
            | 0     | Normal                 |
            +-------+------------------------+
            | 1     | Poisson                |
            +-------+------------------------+
            | (1,2) | Compound Poisson Gamma |
            +-------+------------------------+
            | 2     | Gamma                  |
            +-------+------------------------+
            | 3     | Inverse Gaussian       |
            +-------+------------------------+

            For ``0 < power < 1``, no distribution exists.

    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen `power` parameter as follows:

        - 'identity' for ``power <= 0``, e.g. for the Normal distribution
        - 'log' for ``power > 0``, e.g. for Poisson, Gamma and Inverse Gaussian
          distributions

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PoissonRegressor : Generalized Linear Model with a Poisson distribution.
    GammaRegressor : Generalized Linear Model with a Gamma distribution.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.TweedieRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [2, 3.5, 5, 5.5]
    >>> clf.fit(X, y)
    TweedieRegressor()
    >>> clf.score(X, y)
    0.839...
    >>> clf.coef_
    array([0.599..., 0.299...])
    >>> clf.intercept_
    1.600...
    >>> clf.predict([[1, 1], [3, 4]])
    array([2.500..., 4.599...])
    """

    def __init__(
        self,
        *,
        power=0.0,
        alpha=1.0,
        fit_intercept=True,
        link="auto",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            # Initialize with fixed power. Otherwise, validation would happen in init.
            # In the end, _validate_base_loss returns the used loss.
            _base_loss=HalfTweedieLoss(power=1),
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )
        self.link = link
        self.power = power

    def _validate_base_loss(self):
        # This is only necessary because of the link and power arguments of the
        # TweedieRegressor.
        if not isinstance(self._base_loss, (HalfTweedieLoss, HalfTweedieLossIdentity)):
            raise ValueError(
                "TweedieRegressor._base_loss must be HalfTweedieLoss or"
                " HalfTweedieLossIdentity!"
            )

        # We could raise ValueError. But we want to be able to reset power
        # gracefully.
        # if self.power != self.base_loss.power:
        #     raise ValueError(
        #         "TweedieRegressor must have self.base_loss.power = self.power."
        #     )

        if self.link == "auto":
            if self.power <= 0:
                # identity link
                return HalfTweedieLossIdentity(power=self.power)
            else:
                # log link
                return HalfTweedieLoss(power=self.power)
        elif self.link == "log":
            return HalfTweedieLoss(power=self.power)
        elif self.link == "identity":
            return HalfTweedieLossIdentity(power=self.power)
        else:
            raise ValueError(
                "The link must be an element of ['auto', 'identity', 'log']; "
                f"got (link={self.link!r})"
            )
