# Update tracking issue if Cirrus fails nightly job

# Only update for nightly jobs
echo $CIRRUS_CRON

python -m venv .venv
source .venv/bin/activate
python -m pip install defusedxml PyGithub

LINK_TO_RUN="https://cirrus-ci.com/build/$CIRRUS_BUILD_ID"

echo $LINK_TO_RUN
echo $CIRRUS_REPO_FULL_NAME
echo $CI_NAME

# python -m maint_tools/update_tracking_issue.py \
#    $BOT_GITHUB_TOKEN \
#    $CIRRUS_TASK_NAME \
#    $CIRRUS_REPO_FULL_NAME \
#    $LINK_TO_RUN
#    --tests-passed false
