"""Get version for dependencies."""
import argparse
from sklearn._build_utils.dependencies import package_to_extras

parser = argparse.ArgumentParser(
        description='Get dependencies for from setup.py')
parser.add_argument('--min-version', action='store_true')

# for getting version of a single dependency
parser.add_argument('package', choices=package_to_extras)
args = parser.parse_args()

version_spec = package_to_extras[args.package][0]
if args.min_version:
    version_spec = version_spec.replace(">=", "==")
print("{}{}".format(args.package, version_spec))
