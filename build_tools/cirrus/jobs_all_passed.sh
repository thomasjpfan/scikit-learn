# Get status from the ARM Wheel jobs and return true if they all passed

curl https://api.cirrus-ci.com/v1/artifact/build/$CIRRUS_BUILD_ID/status.zip --output status.zip
unzip status.zip
PASSED_LIST=$(find status -type f -exec cat {} +)

ALL_PASSED="true"
for PASSED in $PASSED_LIST; do
    if [[ "$PASSED" == "false" ]]; then
        ALL_PASSED="false"
        break
    fi
done

echo $ALL_PASSED
