import platform
import re
import site
import sys
from pathlib import Path

PLATFORM_WITH_BUNDLED_LIBRARIES = {"Linux", "Darwin"}
platform_system = platform.system()

if platform_system not in PLATFORM_WITH_BUNDLED_LIBRARIES:
    sys.exit(0)


site_packages = site.getsitepackages()

assert (
    len(site_packages) == 1
), "There should be only one site packages in test environment"

site_packages_path = Path(site_packages[0])

try:
    distinfo_path = next(s for s in site_packages_path.glob("scikit_learn-*.dist-info"))
except StopIteration as e:
    raise RuntimeError("Unable to find scikit-learn's dist-info") from e

license_text = (distinfo_path / "COPYING").read_text()

assert "Copyright (c)" in license_text

found = re.search(
    r"This binary distribution of \w+ also bundles the following software", license_text
)
assert found, f"Unable to find bundled license for {platform.system()}"
