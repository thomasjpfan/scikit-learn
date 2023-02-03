# This script uses starlark for configuring when a cirrus CI job runs:
# https://cirrus-ci.org/guide/programming-tasks/

load("cirrus", "env", "fs", "http")

def main(ctx):
    # Only run for scikit-learn/scikit-learn. For debugging on a fork, you can
    # comment out the following condition.
    arm_tests_yaml = "build_tools/cirrus/arm_tests.yml"

    return fs.read(arm_tests_yaml)
