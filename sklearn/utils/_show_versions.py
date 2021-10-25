"""
Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""
# License: BSD 3 clause

import platform
import sys
import importlib
from threadpoolctl import threadpool_info


from ._openmp_helpers import _openmp_parallelism_enabled


def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """Overview of the installed version of main dependencies

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    deps = [
        "pip",
        "setuptools",
        "sklearn",
        "numpy",
        "scipy",
        "Cython",
        "pandas",
        "matplotlib",
        "joblib",
        "threadpoolctl",
    ]

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Print useful debugging information"

    .. versionadded:: 0.20
    """

    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))

    built_with_openmp = _openmp_parallelism_enabled()
    print("\n{k}: {stat}".format(k="Built with OpenMP", stat=built_with_openmp))

    # show threadpoolctl results when built with openmp
    if built_with_openmp:
        threadpool_results = threadpool_info()
        if threadpool_results:
            print()
            print("threadpoolctl info:")
            keys = [
                "filepath",
                "prefix",
                "user_api",
                "internal_api",
                "version",
                "num_threads",
            ]
            display_format = "{key:>13}: {value}"
            for i, result in enumerate(threadpool_results):
                for key in keys:
                    print(display_format.format(key=key, value=result[key]))
                if i != len(threadpool_results) - 1:
                    print()
