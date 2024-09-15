import platform
import re
import site
from itertools import chain
from pathlib import Path

site_packages = site.getsitepackages()

site_packages_path = (Path(p) for p in site_packages)

try:
    distinfo_path = next(
        chain(
            s
            for site_package in site_packages_path
            for s in site_package.glob("scikit_learn-*.dist-info")
        )
    )
except StopIteration as e:
    raise RuntimeError("Unable to find scikit-learn's dist-info") from e

license_text = (distinfo_path / "COPYING").read_text()

assert "Copyright (c)" in license_text

found = re.search(
    r"This binary distribution of \w+ also bundles the following software", license_text
)
assert found, f"Unable to find bundled license for {platform.system()}"
