# This script uses starlark for configuring when a cirrus CI job runs:
# https://cirrus-ci.org/guide/programming-tasks/

load("cirrus", "env", "fs", "http")

def main(ctx):
    # Only run for scikit-learn/scikit-learn. For debugging on a fork, you can
    # comment out the following condition.
    #if env.get("CIRRUS_REPO_FULL_NAME") != "scikit-learn/scikit-learn":
    #    return []

    arm_wheel_yaml = "build_tools/cirrus/arm_wheel.yml"
    return fs.read(arm_wheel_yaml)
