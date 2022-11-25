# This script uses starlark for configuring when a cirrus CI job runs:
# https://cirrus-ci.org/guide/programming-tasks/

load("cirrus", "env", "fs", "http")

def main(ctx):
    # Get commit message for event.
    SHA = env.get("CIRRUS_CHANGE_IN_REPO")
    url = "https://api.github.com/repos/scipy/scipy/git/commits/" + SHA
    dct = http.get(url).json()
    if "[wheel build]" in dct["message"]:
        return fs.read("ci/cirrus_wheels.yml")

    if "[skip ci]" in dct["message"]:
        return []
    return fs.read("build_tools/cirrus/arm_ci.yml")
