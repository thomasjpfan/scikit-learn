"""Global configuration state and functions for management
"""
import os
from contextlib import contextmanager as contextmanager
import threading

_global_config = {
    "assume_finite": bool(os.environ.get("SKLEARN_ASSUME_FINITE", False)),
    "working_memory": int(os.environ.get("SKLEARN_WORKING_MEMORY", 1024)),
    "print_changed_only": True,
    "display": "diagram",
    "pairwise_dist_chunk_size": int(
        os.environ.get("SKLEARN_PAIRWISE_DIST_CHUNK_SIZE", 256)
    ),
    "enable_cython_pairwise_dist": True,
    "array_api_dispatch": False,
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    set_config : Set global scikit-learn configuration.
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()


def set_config(
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
    pairwise_dist_chunk_size=None,
    enable_cython_pairwise_dist=None,
    array_api_dispatch=None,
):
    """Set global scikit-learn configuration

    .. versionadded:: 0.19

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

        .. versionadded:: 0.19

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

        .. versionadded:: 0.20

    print_changed_only : bool, default=None
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()' while the default
        behaviour would be to print 'SVC(C=1.0, cache_size=200, ...)' with
        all the non-changed parameters.

        .. versionadded:: 0.21

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. Default is 'diagram'.

        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        The number of row vectors per chunk for the accelerated pairwise-
        distances reduction backend. Default is 256 (suitable for most of
        modern laptops' caches and architectures).

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        Use the accelerated pairwise-distances reduction backend when
        possible. Global default: True.

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        .. versionadded:: 1.1

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.
    """
    local_config = _get_threadlocal_config()

    if assume_finite is not None:
        local_config["assume_finite"] = assume_finite
    if working_memory is not None:
        local_config["working_memory"] = working_memory
    if print_changed_only is not None:
        local_config["print_changed_only"] = print_changed_only
    if display is not None:
        local_config["display"] = display
    if pairwise_dist_chunk_size is not None:
        local_config["pairwise_dist_chunk_size"] = pairwise_dist_chunk_size
    if enable_cython_pairwise_dist is not None:
        local_config["enable_cython_pairwise_dist"] = enable_cython_pairwise_dist
    if array_api_dispatch is not None:
        local_config["array_api_dispatch"] = array_api_dispatch


@contextmanager
def config_context(
    *,
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
    pairwise_dist_chunk_size=None,
    enable_cython_pairwise_dist=None,
    array_api_dispatch=None,
):
    """Context manager for global scikit-learn configuration.

    Parameters
    ----------
    assume_finite : bool, default=None
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error. If None, the existing value won't change.
        The default value is False.

    working_memory : int, default=None
        If set, scikit-learn will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. If None, the existing value won't change.
        The default value is 1024.

    print_changed_only : bool, default=None
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()', but would print
        'SVC(C=1.0, cache_size=200, ...)' with all the non-changed parameters
        when False. If None, the existing value won't change.
        The default value is True.

        .. versionchanged:: 0.23
           Default changed from False to True.

    display : {'text', 'diagram'}, default=None
        If 'diagram', estimators will be displayed as a diagram in a Jupyter
        lab or notebook context. If 'text', estimators will be displayed as
        text. If None, the existing value won't change.
        The default value is 'diagram'.

        .. versionadded:: 0.23

    pairwise_dist_chunk_size : int, default=None
        The number of row vectors per chunk for the accelerated pairwise-
        distances reduction backend. Default is 256 (suitable for most of
        modern laptops' caches and architectures).

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    enable_cython_pairwise_dist : bool, default=None
        Use the accelerated pairwise-distances reduction backend when
        possible. Global default: True.

        Intended for easier benchmarking and testing of scikit-learn internals.
        End users are not expected to benefit from customizing this configuration
        setting.

        .. versionadded:: 1.1

    array_api_dispatch : bool, default=None
        Use Array API dispatching when inputs follow the Array API standard.
        Default is False.

        See the :ref:`User Guide <array_api>` for more details.

        .. versionadded:: 1.1

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import sklearn
    >>> from sklearn.utils.validation import assert_all_finite
    >>> with sklearn.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with sklearn.config_context(assume_finite=True):
    ...     with sklearn.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN...
    """
    old_config = get_config()
    set_config(
        assume_finite=assume_finite,
        working_memory=working_memory,
        print_changed_only=print_changed_only,
        display=display,
        pairwise_dist_chunk_size=pairwise_dist_chunk_size,
        enable_cython_pairwise_dist=enable_cython_pairwise_dist,
        array_api_dispatch=array_api_dispatch,
    )

    try:
        yield
    finally:
        set_config(**old_config)
