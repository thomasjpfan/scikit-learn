"""All dependencies for scikit-learn."""
from typing import Dict, Set
from collections import defaultdict
import platform


# numpy scipy and cython should by in sync with pyproject.toml
if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.19.1'
    NUMPY_MIN_VERSION = '1.13.3'

CYTHON_MIN_VERSION = '0.28.5'
JOBLIB_MIN_VERSION = '0.11'
THREADPOOLCTL_MIN_VERSION = '2.0.0'
PYTEST_MIN_VERSION = '3.3.0'


# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma seperated tags)
package_to_extras = {
    'numpy': ('>={}'.format(NUMPY_MIN_VERSION), 'build,install'),
    'scipy': ('>={}'.format(SCIPY_MIN_VERSION), 'build,install'),
    'joblib': ('>={}'.format(JOBLIB_MIN_VERSION), 'install'),
    'threadpoolctl': ('>={}'.format(THREADPOOLCTL_MIN_VERSION), 'install'),
    'cython': ('>={}'.format(CYTHON_MIN_VERSION), 'build'),
    'matplotlib': ('>=2.1.1', 'benchmark,docs,examples,tests'),
    'scikit-image': ('>=0.13', 'docs,examples,tests'),
    'pandas': ('>=0.18.0', 'benchmark,docs,examples,tests'),
    'seaborn': ('>=0.9.0', 'docs,examples'),
    'memory_profiler': ('>=0.57.0', 'benchmark,docs'),
    'pytest': ('>={}'.format(PYTEST_MIN_VERSION), 'tests'),
    'pytest-xdist': ('>=1.32.0', 'tests'),
    'pytest-cov': ('>=2.9.0', 'tests'),
    'flake8': ('>=3.8.2', 'tests'),
    'mypy': ('>=0.770', 'tests'),
    'pyamg': ('>=4.0.0', 'tests'),
    'sphinx': ('>=2.1.2', 'docs'),
    'sphinx-gallery>=0.7.0': ('>=0.7.0', 'docs'),
    'numpydoc': ('>=0.9.2', 'docs'),
    'Pillow': ('>=7.1.2', 'docs'),
}


# create inverse mapping for setuptools
extras_to_requires: Dict[str, Set[str]] = defaultdict(set)
for package, (version_spec, extras) in package_to_extras.items():
    for extra in extras.split(','):
        extras_to_requires[extra].add("{}{}".format(package, version_spec))
