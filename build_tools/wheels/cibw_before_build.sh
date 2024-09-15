#!/bin/bash

set -euxo pipefail

PROJECT_DIR="$1"
LICENSE_FILE="$PROJECT_DIR/COPYING"

cat <<EOF >>$LICENSE_FILE

----

EOF

if [[ $RUNNER_OS == "Linux" ]]; then
    cat $PROJECT_DIR/build_tools/wheels/LICENSE_linux.txt >>$LICENSE_FILE
elif [[ $RUNNER_OS == "macOS" ]]; then
    cat $PROJECT_DIR/build_tools/wheels/LICENSE_osx.txt >>$LICENSE_FILE
fi
