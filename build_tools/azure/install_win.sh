#!/bin/bash
set -ex

if [ "$PYTHON_ARCH" == "64" ]; then
    conda create -n $VIRTUALENV -q -y python=$PYTHON_VERSION numpy scipy cython matplotlib wheel pillow joblib

    source activate $VIRTUALENV
    pip install threadpoolctl
else
    pip install numpy scipy cython wheel pillow joblib threadpoolctl
fi

if [ "$PYTEST_VERSION" == "*" ]; then
    pip install pytest
else
    pip install pytest==$PYTEST_VERSION
fi

if [ "$COVERAGE" == "true" ]; then
    pip install pytest-cov codecov
fi

python --version
pip --version

# Install the build and runtime dependencies of the project.
python setup.py bdist_wheel bdist_wininst -b doc/logos/scikit-learn-logo.bmp

# Install the generated wheel package to test it
pip install --pre --pre --no-index --find-links dist/ scikit-learn
