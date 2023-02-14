# Update tracking issue if Cirrus fails nightly job

curl https://api.cirrus-ci.com/v1/artifact/build/$CIRRUS_BUILD_ID/status.zip --output status.zip
unzip status.zip
PASSED_LIST=$(find status -type f -exec cat {} +)

echo $PASSED_LIST

ALL_PASSED="true"
for PASSED in $PASSED_LIST; do
   if [[ "$PASSED" == "false" ]]; then
      ALL_PASSED="false"
      break
   fi
done

echo $ALL_PASSED

# python -m venv .venv
# source .venv/bin/activate
# python -m pip install defusedxml PyGithub

# LINK_TO_RUN="https://cirrus-ci.com/build/$CIRRUS_BUILD_ID"

# python maint_tools/update_tracking_issue.py \
#    $BOT_GITHUB_TOKEN \
#    "ARM Wheels" \
#    $CIRRUS_REPO_FULL_NAME \
#    $LINK_TO_RUN \
#    --tests-passed $ALL_PASSED

# if [[ "$ALL_PASSED" == "true" ]]; then
#    exit 0
# else
#    exit 1
# fi
