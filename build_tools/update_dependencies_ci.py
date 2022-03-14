import argparse
import re
import sys
from ruamel.yaml import YAML
import json
from urllib import request


def check_pypa(package, current_version):
    response = request.urlopen(f"https://pypi.org/pypi/{package}/json")
    response_json = json.load(response)
    latest_version = response_json["info"]["version"]
    if current_version != latest_version:
        return latest_version
    return current_version


def get_version(dependency):
    # get package name
    match = re.search(r"(\w+)", dependency)
    if not match:
        raise ValueError("No package found")

    package = match.group(1)

    # check for version name
    current_version = ""
    full_match = re.search(r"(\w+)[=<>]+([\d\.]+)", dependency)
    if full_match:
        current_version = full_match.group(2)

    return (package, current_version)


parser = argparse.ArgumentParser(description="Update environment yaml")
parser.add_argument(
    "env_file", help="conda environment file", type=argparse.FileType("r")
)
parser.add_argument("conda_env", help="conda env to update")

args = parser.parse_args()
conda_env = args.conda_env

yaml = YAML()
env_file = yaml.load(args.env_file)
comments = env_file["dependencies"].ca.items


need_update = False
pip_dependencies = None


# Check conda dependencies
for i, dependency in enumerate(env_file["dependencies"]):
    if isinstance(dependency, dict) and "pip" in dependency:
        # store pip dependency for later use
        pip_dependencies = dependency["pip"]
        continue

    comment = comments.get(i)
    if comment is None or "latest" not in comment[0].value:
        # Only update entries with a "latest" comment
        continue

    try:
        package, current_version = get_version(dependency)
    except ValueError:
        continue

    new_version = check_pypa(package, current_version)
    if new_version != current_version:
        need_update = True
        env_file["dependencies"][i] = f"{package}<={new_version}"


# Check pip dependencies
if pip_dependencies is not None:
    pip_comments = pip_dependencies.ca.items
    for i, dependency in enumerate(pip_dependencies):
        comment = comments.get(i)
        if comment is None or "latest" not in comment[0].value:
            # Only update entries with a "latest" comment
            continue

        try:
            package, current_version = get_version(dependency)
        except ValueError:
            continue

        new_version = check_pypa(package, current_version)
        if new_version != current_version:
            need_update = True
            pip_dependencies[i] = f"{package}<={new_version}"

if need_update:
    yaml.dump(env_file, sys.stdout)
