#!/bin/bash

set -euxo pipefail

PROJECT_DIR="$1"
LICENSE_FILE="$PROJECT_DIR/COPYING"

echo "" >> $PROJECT_DIR/LICENSE.txt
echo "----" >> $PROJECT_DIR/LICENSE.txt
echo "" >> $PROJECT_DIR/LICENSE.txt

if [[ $RUNNER_OS == "Linux" ]]; then
    cat $PROJECT_DIR/build_tools/wheels/LICENSE_linux.txt >>$LICENSE_FILE
elif [[ $RUNNER_OS == "macOS" ]]; then
    cat $PROJECT_DIR/build_tools/wheels/LICENSE_macos.txt >>$LICENSE_FILE
elif [[ $RUNNER_OS == "Windows" ]]; then
    cat $PROJECT_DIR/build_tools/wheels/LICENSE_windows.txt >>$LICENSE_FILE
fi
