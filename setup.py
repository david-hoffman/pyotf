#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py
"""
Setup files

Copyright (c) 2020, David Hoffman
"""

import setuptools

# read in long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# get requirements
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="pyotf",
    version="0.0.1",
    author="David Hoffman",
    author_email="dave.p.hoffman@gmail.com",
    description="A python library for simulating and analyzing microscope point spread functions (PSFs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3",
    install_requires=requirements,
)
