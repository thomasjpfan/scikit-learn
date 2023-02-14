# Update tracking issue if Cirrus fails nightly job

curl https://api.cirrus-ci.com/v1/artifact/build/$CIRRUS_BUILD_ID/passed.zip --output passed.zip
unzip passed.zip
PASSED=$(cat passed.txt)

python -m pip install defusedxml PyGithub

LINK_TO_RUN="https://cirrus-ci.com/build/$CIRRUS_BUILD_ID"

python maint_tools/update_tracking_issue.py \
   $BOT_GITHUB_TOKEN \
   "ARM Wheels" \
   $CIRRUS_REPO_FULL_NAME \
   $LINK_TO_RUN \
   --tests-passed false

if [[ "$PASSED" == "true" ]]; then
   exit 0
else
   exit 1
fi
