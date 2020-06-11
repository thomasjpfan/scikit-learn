"""Get version for dependencies. Used only in the CI"""
import sys
from pathlib import Path
import argparse

# make sklearn._build_utils avaliable
sys.path.insert(0, str(Path('.').parent))

from dependencies import package_to_extras  # noqa

parser = argparse.ArgumentParser(
        description='Get dependencies for from setup.py')
parser.add_argument('--min-version', action='store_true')

# for getting version of a single dependency
parser.add_argument('package', choices=package_to_extras)
args = parser.parse_args()

version_spec = package_to_extras[args.package][0]
if args.min_version:
    version_spec = version_spec.replace(">=", "==")

# package==version or package>=version
print("{}{}".format(args.package, version_spec))
