# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension, Distribution
from os import path, environ
from shutil import copyfile
from glob import glob

app_name = "pyobjdetect"
app_version = "0.0.0"
license_filename = "LICENSE.txt"
readme_md_file = "readme.md"
version_file = "_version.txt"
description_package = "Object detection using Deep Learning"

try:
    with open(version_file) as f:
        app_version = f.read()
except:
    pass


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(foo):
        return True


setup(
    name=app_name,
    version=app_version,
    author="Neeth Kunnath",
    author_email="neeth.xavier@gmail.com",
    license=license_filename,
    description=description_package,
    packages=find_packages(),
    distclass=BinaryDistribution,
    package_data={"": ["*.pyd"]},
    # only for source dist + Manifest.in
    include_package_data=True,
    url="",
    # python = 3.10
    # pip install packages dependencies
    install_requires=[
        "wheel",
    ],
    extras_require={"dev": ["black"]},
    entry_points={
        "console_scripts": [
        ]
    },
)
