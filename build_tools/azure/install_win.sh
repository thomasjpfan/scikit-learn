#!/bin/bash

set -e
set -x

# if [[ "$PYTHON_ARCH" == "64" ]]; then
#     # conda create -n $VIRTUALENV -q -y python=$PYTHON_VERSION numpy scipy cython

#     # source activate $VIRTUALENV

#     pip install threadpoolctl scipy numpy cython

#     if [[ "$PYTEST_VERSION" == "*" ]]; then
#         pip install pytest
#     else
#         pip install pytest==$PYTEST_VERSION
#     fi
# else
#     pip install numpy scipy cython pytest wheel pillow joblib threadpoolctl
# fi

# if [[ "$PYTEST_XDIST_VERSION" != "none" ]]; then
#     pip install pytest-xdist
# fi

# if [[ "$COVERAGE" == "true" ]]; then
#     pip install coverage codecov pytest-cov
# fi

# python --version
# pip --version

# pip install scikit-learn

# Build scikit-learn
# python setup.py bdist_wheel

# Install the generated wheel package to test it
# pip install --pre --no-index --find-links dist scikit-learn

TEST_CMD="python -m pytest --showlocals --durations=20"

COVERAGE_PROCESS_START="$BUILD_SOURCESDIRECTORY/.coveragerc"
TEST_CMD="$TEST_CMD --cov-config=$COVERAGE_PROCESS_START --cov sklearn --cov-report="

# Python 3.10 deprecates disutils and is imported by numpy interally during import time
PYTHON_3_10_WARNING="-Wignore:The\ distutils:DeprecationWarning"
TEST_CMD="$TEST_CMD $PYTHON_3_10_WARNING"

eval "$TEST_CMD"
