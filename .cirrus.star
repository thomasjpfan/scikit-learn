# This script uses starlark for configuring when a cirrus CI job runs:
# https://cirrus-ci.org/guide/programming-tasks/

load("cirrus", "env", "fs", "http")

def main(ctx):
    return fs.read("build_tools/cirrus/arm_ci.yml")
