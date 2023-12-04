# Authors: Liam Arguedas <iliamftw2013@gmail.com>
# License: BSD 3 clause

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "extralearning"
DESCRIPTION = "extralearning is a comprehensive package that consolidates leading ML frameworks and introduces functionality to streamline code in ML projects."
URL = "https://github.com/extra-learning/extralearning"
EMAIL = "iliamftw2013@gmail.com"
AUTHOR = "Liam Arguedas"
REQUIRES_PYTHON = ">=3.8.0"

# Long description
with open("README.md", "r") as file:
    LONG_DESCRIPTION = file.read()


# requirements.txt: Packages required for this module to be executed
def read_requirements(filename="requirements.txt"):
    with open(filename) as file:
        return file.read().splitlines()


# Load the package's VERSION file as a dictionary.
about = {}

# Directory cfgs
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "extralearning"

with open(PACKAGE_DIR / "VERSION") as file:
    VERSION = file.read().strip()
    about["__version__"] = VERSION

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    package_data={"extralearning": ["VERSION", "py.typed"]},
    license="BSD 3 clause",
    install_requires=read_requirements(),
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
