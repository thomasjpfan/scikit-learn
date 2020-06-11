"""All dependencies for scikit-learn."""
from collections import defaultdict
import platform
import argparse


# numpy scipy and cython should by in sync with pyproject.toml
if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.19.1'
    NUMPY_MIN_VERSION = '1.13.3'

JOBLIB_MIN_VERSION = '0.11'
THREADPOOLCTL_MIN_VERSION = '2.0.0'
PYTEST_MIN_VERSION = '3.3.0'

# Keep in sync with sklearn/_build_utils/__init__.py
CYTHON_MIN_VERSION = '0.28.5'


# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma seperated tags)
package_to_extras = {
    'numpy': (NUMPY_MIN_VERSION, 'build,install'),
    'scipy': (SCIPY_MIN_VERSION, 'build,install'),
    'joblib': (JOBLIB_MIN_VERSION, 'install'),
    'threadpoolctl': (THREADPOOLCTL_MIN_VERSION, 'install'),
    'cython': (CYTHON_MIN_VERSION, 'build'),
    'matplotlib': ('2.1.1', 'benchmark,docs,examples,tests'),
    'scikit-image': ('0.13', 'docs,examples,tests'),
    'pandas': ('0.18.0', 'benchmark,docs,examples,tests'),
    'seaborn': ('0.9.0', 'docs,examples'),
    'memory_profiler': ('0.57.0', 'benchmark,docs'),
    'pytest': (PYTEST_MIN_VERSION, 'tests'),
    'pytest-xdist': ('1.32.0', 'tests'),
    'pytest-cov': ('2.9.0', 'tests'),
    'flake8': ('3.8.2', 'tests'),
    'mypy': ('0.770', 'tests'),
    'pyamg': ('4.0.0', 'tests'),
    'sphinx': ('2.1.2', 'docs'),
    'sphinx-gallery0.7.0': ('0.7.0', 'docs'),
    'numpydoc': ('0.9.2', 'docs'),
    'Pillow': ('7.1.2', 'docs'),
}


# create inverse mapping for setuptools
extras_to_requires: dict = defaultdict(set)
for package, (min_version, extras) in package_to_extras.items():
    for extra in extras.split(','):
        extras_to_requires[extra].add("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Get dependencies for from setup.py')

    # for getting version of a single dependency
    parser.add_argument('package', choices=package_to_extras)
    args = parser.parse_args()
    min_version = package_to_extras[args.package][0]

    print(min_version)
