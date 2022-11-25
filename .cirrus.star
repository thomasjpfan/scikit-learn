# This script uses starlark for configuring when a cirrus CI job runs:
# https://cirrus-ci.org/guide/programming-tasks/

load("cirrus", "env", "fs", "http")

def main(ctx):
    # Only run for scikit-learn/scikit-learn. For debugging on a fork, you can
    # comment out the following condition.
    if env.get("CIRRUS_REPO_FULL_NAME") != "scikit-learn/scikit-learn":
        return []

    # Get commit message for event.
    SHA = env.get("CIRRUS_CHANGE_IN_REPO")
    url = "https://api.github.com/repos/scikit-learn/scikit-learn/git/commits/" + SHA
    response = http.get(url).json()
    commit_message = response["message"]

    if "[skip ci]" in commit_message:
        return []
    return fs.read("build_tools/cirrus/arm_ci.yml")
