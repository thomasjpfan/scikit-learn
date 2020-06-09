#!/bin/bash
set -ex

if [ "$PYTHON_ARCH" == "64" ]; then
    conda create -n $VIRTUALENV -q -y python=$PYTHON_VERSION numpy scipy cython matplotlib wheel pillow joblib

    source activate $VIRTUALENV
    python -m pip install threadpoolctl
else
    python -m pip install numpy scipy cython wheel pytest pillow joblib threadpoolctl
fi

if [ "$PYTEST_VERSION" == "*" ]; then
    python -m pip install pytest
else
    python -m pip install pytest==$PYTEST_VERSION
fi
python -m pip install pytest-xdist

if [ "$COVERAGE" == "true" ]; then
    python -m pip install pytest-cov codecov
fi

python --version
python -m pip --version

# Install the build and runtime dependencies of the project.
python setup.py bdist_wheel bdist_wininst -b doc/logos/scikit-learn-logo.bmp

# Install the generated wheel package to test it
python -m pip install --pre --pre --no-index --find-links dist/ scikit-learn
