#!/bin/bash

set -e

if [[ "$DISTRIB" =~ ^conda ]]; then
    export PATH=$HOME/miniconda3/bin:$PATH
    source activate $VIRTUALENV
else
    source $VIRTUALENV/bin/activate
fi

# Need to run codecov from a git checkout, so we copy .coverage
# from TEST_DIR where pytest has been run
pushd $TEST_DIR
coverage combine --append
popd
cp $TEST_DIR/.coverage $BUILD_REPOSITORY_LOCALPATH

codecov --root $BUILD_REPOSITORY_LOCALPATH -t $CODECOV_TOKEN || echo "codecov upload failed"
