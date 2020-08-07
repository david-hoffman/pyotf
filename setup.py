#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py
"""
Setup files

Copyright (c) 2020, David Hoffman
"""

import setuptools
import versioneer

# read in long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# get requirements
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="pyotf",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="David Hoffman",
    author_email="dave.p.hoffman@gmail.com",
    url="https://github.com/david-hoffman/pyOTF",
    description="A python library for simulating and analyzing microscope point spread functions (PSFs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache V2.0",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
