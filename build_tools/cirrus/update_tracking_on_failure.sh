# Update tracking issue if Cirrus fails nightly job

ALL_PASSED=$(bash build_tools/cirrus/jobs_all_passed.sh)

python -m venv .venv
source .venv/bin/activate
python -m pip install defusedxml PyGithub

LINK_TO_RUN="https://cirrus-ci.com/build/$CIRRUS_BUILD_ID"

python maint_tools/update_tracking_issue.py \
    $BOT_GITHUB_TOKEN \
    "ARM Wheels" \
    $CIRRUS_REPO_FULL_NAME \
    $LINK_TO_RUN \
    --tests-passed $ALL_PASSED

if [[ "$ALL_PASSED" == "true" ]]; then
    exit 0
else
    exit 1
fi
