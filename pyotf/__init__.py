#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
A package to simulate optical transfer functions and point spread functions
and perform phase retrieval on experimental data.

https://en.wikipedia.org/wiki/Optical_transfer_function
https://en.wikipedia.org/wiki/Point_spread_function
https://en.wikipedia.org/wiki/Phase_retrieval

Copyright (c) 2016, David Hoffman
"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
