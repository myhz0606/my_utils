#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from setuptools import setup, find_packages


try:
    README = open('README.md').read()
except Exception:
    README = ""
VERSION = "0.0.7"

requirments = ["numpy",
               "lmdb",
               "Pillow",
               "scipy"]

setup(
    name='my_utils',
    version=VERSION,
    description='some utils function',
    url="git@github.com:myhz0606/my_utils.git",
    long_description=README,
    author='myhz0606',
    author_email='myhz0606@gmail.com',
    packages=find_packages(),
    install_requires=requirments,
    include_package_data=True, 
    extras_require={
        # "extra": ["extra_requirments"],
    },
    entry_points={
        # 'console_scripts': [
        #     'modelhub=modelhub.commands:main'
        # ]
    },
)