#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

# Travis clone scikit-learn/scikit-learn repository in to a local repository.
# We use a cached directory with three scikit-learn repositories (one for each
# matrix entry) from which we pull from local Travis repository. This allows
# us to keep build artefact for gcc + cython, and gain time

set -e

# arm64 does not have ccache or jq by default
if [ $TRAVIS_CPU_ARCH == "arm64" ]; then
    sudo apt-get install ccache jq
fi

# Fail fast
build_tools/travis/travis_fastfail.sh

echo "List files from cached directories"
echo "pip:"
ls $HOME/.cache/pip

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

dev_url=https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com
if [ $TRAVIS_CPU_ARCH == "amd64" ]; then
    pip install --upgrade pip setuptools
    echo "Installing numpy and scipy master wheels"
    pip install --pre --upgrade --timeout=60 -f $dev_url numpy scipy pandas cython
    echo "Installing pillow master"
    pip install https://github.com/python-pillow/Pillow/archive/master.zip
elif [ $TRAVIS_CPU_ARCH == "arm64" ]; then
    sudo apt-get install python3-dev python3-scipy python3-numpy python3-pip python3-pandas python3-pil python3-virtualenv
    python3 -m virtualenv --system-site-packages --python=python3 testenv

    source testenv/bin/activate
    pip install cython
fi

echo "Installing joblib master"
pip install https://github.com/joblib/joblib/archive/master.zip
pip install pytest==4.6.4 pytest-cov

# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [ $TRAVIS_CPU_ARCH == "amd64" ]; then
    pip install -v -e .
elif [ $TRAVIS_CPU_ARCH == "arm64" ]; then
    pip install -v -e . --user
fi


ccache --show-stats
# Useful for debugging how ccache is used
# cat $CCACHE_LOGFILE

# fast fail
build_tools/travis/travis_fastfail.sh
