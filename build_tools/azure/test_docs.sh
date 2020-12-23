#!/bin/bash

set -e

if [[ "$DISTRIB" =~ ^conda.* ]]; then
    source activate $VIRTUALENV
elif [[ "$DISTRIB" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

if [[ "$DISTRIB" == "conda-pip-icc-build" ]]; then
    source /opt/intel/oneapi/compiler/latest/env/vars.sh
fi

make test-doc
