#!/bin/bash

set -e

# defines the show_installed_libraries function
source build_tools/shared.sh

activate_environment

# Need to run codecov from a git checkout, so we copy .coverage
# from TEST_DIR where pytest has been run
pushd $TEST_DIR
coverage combine --append
popd
cp $TEST_DIR/.coverage $BUILD_REPOSITORY_LOCALPATH
