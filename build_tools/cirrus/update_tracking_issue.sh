# Update tracking issue if Cirrus fails nightly job

echo $CIRRUS_CRON
python -m venv .venv
source .venv/bin/activate
python -m pip install defusedxml PyGithub

LINK_TO_RUN="https://cirrus-ci.com/build/$CIRRUS_BUILD_ID"
ISSUE_REPO="$CIRRUS_REPO_FULL_NAME"
CI_NAME="$1"

echo $LINK_TO_RUN
echo $CIRRUS_REPO_FULL_NAME
echo $CI_NAME
