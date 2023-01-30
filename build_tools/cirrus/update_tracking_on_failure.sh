# Update tracking issue if Cirrus fails nightly job

# Only update for nightly jobs and when the bot token is set
echo $CIRRUS_CRON
if [[ -z $BOT_GITHUB_TOKEN ]]; then
   exit 0
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install defusedxml PyGithub

LINK_TO_RUN="https://cirrus-ci.com/build/$CIRRUS_BUILD_ID"

python maint_tools/update_tracking_issue.py \
   $BOT_GITHUB_TOKEN \
   $CIRRUS_TASK_NAME \
   $CIRRUS_REPO_FULL_NAME \
   $LINK_TO_RUN \
   --tests-passed false
