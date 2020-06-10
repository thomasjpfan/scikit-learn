"""Get version for dependencies."""
import argparse
from sklearn._build_utils.dependencies import extras_to_requires
from sklearn._build_utils.dependencies import package_to_extras


parser = argparse.ArgumentParser(
        description='Get dependencies for from setup.py')
parser.add_argument('--min-version', action='store_true')
subparser = parser.add_subparsers(dest='subparser_name')

# for getting versions for all dependencies based on extra_requires
extra_parser = subparser.add_parser('extra_requires',
                                    help='get version for extra requires')
extra_parser.add_argument('extra_requires', choices=extras_to_requires.keys())

# for getting version of a single dependency
single_parser = subparser.add_parser('single',
                                     help='get version for a single')
single_parser.add_argument('single', choices=package_to_extras)
args = parser.parse_args()

if args.subparser_name == 'extra_requires':
    dependencies = extras_to_requires[args.extra_requires]
    if args.min_version:
        dependencies = [dep.replace(">=", "==") for dep in dependencies]
    print(" ".join(dependencies))
else:  # args.subparser_name == single
    version_spec = package_to_extras[args.single][0]
    if args.min_version:
        version_spec = version_spec.replace(">=", "==")
    print("{}{}".format(args.single, version_spec))
