#!/bin/bash

set -e

UNAMESTR=`uname`

if [[ "$UNAMESTR" == "Darwin" ]]; then
    # install OpenMP not present by default on osx
    HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp

    # enable OpenMP support for Apple-clang
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
    export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

    # avoid error due to multiple OpenMP libraries loaded simultaneously
    export KMP_DUPLICATE_LIB_OK=TRUE
fi

make_conda() {
    TO_INSTALL="$@"
    conda create -n $VIRTUALENV --yes $TO_INSTALL
    source activate $VIRTUALENV
}

if [[ "$DISTRIB" == "conda" ]]; then
    TO_INSTALL="python=$PYTHON_VERSION pip pytest pytest-cov \
                numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
                cython=$CYTHON_VERSION"

    if [[ "$INSTALL_MKL" == "true" ]]; then
        TO_INSTALL="$TO_INSTALL mkl"
    else
        TO_INSTALL="$TO_INSTALL nomkl"
    fi

    if [[ -n "$PANDAS_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pandas=$PANDAS_VERSION"
    fi

    if [[ -n "$PYAMG_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pyamg=$PYAMG_VERSION"
    fi

    if [[ -n "$PILLOW_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL pillow=$PILLOW_VERSION"
    fi

    if [[ -n "$JOBLIB_VERSION" ]]; then
        TO_INSTALL="$TO_INSTALL joblib=$JOBLIB_VERSION"
    fi

	make_conda $TO_INSTALL

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo apt-get install python3-scipy libatlas3-base libatlas-base-dev libatlas-dev python3-virtualenv
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
    python -m pip install pytest pytest-cov cython joblib==$JOBLIB_VERSION

elif [[ "$DISTRIB" == "scipy-dev" ]]; then
    make_conda python=$PYTHON_VERSION
    python -m pip install --upgrade pip setuptools

    echo "Installing numpy and scipy master wheels"
    dev_url=https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com
    python -m pip install --pre --upgrade --timeout=60 -f $dev_url numpy scipy pandas cython
    echo "Installing joblib master"
    python -m pip install https://github.com/joblib/joblib/archive/master.zip
    export SKLEARN_SITE_JOBLIB=1
    echo "Installing pillow master"
    python -m pip install https://github.com/python-pillow/Pillow/archive/master.zip
    python -m pip install pytest pytest-cov

fi

if [[ "$COVERAGE" == "true" ]]; then
    python -m pip install coverage codecov
fi

if [[ "$TEST_DOCSTRINGS" == "true" ]]; then
    python -m pip install sphinx numpydoc  # numpydoc requires sphinx
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "\
try:
    import pandas
    print('pandas %s' % pandas.__version__)
except ImportError:
    print('pandas not installed')
"
pip list
python setup.py develop
