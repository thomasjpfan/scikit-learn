#!/bin/bash

set -e
set -x

# OpenMP is not present on macOS by default
if [[ "$RUNNER_OS" == "macOS" ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # scikit-learn wheels for macos. The list of binaries are in
    # https://packages.macports.org/libomp/.
    git clone https://github.com/thomasjpfan/libomp-osx-artifacts --depth 1
    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        # arm64 builds must cross compile because CI is on x64
        export PYTHON_CROSSENV=1
        # SciPy requires 12.0 on arm to prevent kernel panics
        # https://github.com/scipy/scipy/issues/14688
        # We use the same deployment target to match SciPy.
        export MACOSX_DEPLOYMENT_TARGET=12.0
        ROOT_FOLDER=$PWD/libomp-osx-artifacts/11.0.1/arm64/
    else
        export MACOSX_DEPLOYMENT_TARGET=10.13
        ROOT_FOLDER=$PWD/libomp-osx-artifacts/11.0.1/x86_64/
    fi

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I$ROOT_FOLDER/include"
    export CXXFLAGS="$CXXFLAGS -I$ROOT_FOLDER/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,$ROOT_FOLDER/lib -L$ROOT_FOLDER/lib -lomp"
fi

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
