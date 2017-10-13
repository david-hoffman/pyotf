#!/usr/bin/env python
# -*- coding: utf-8 -*-
# phaseretrieval.py
"""
Back focal plane (pupil) phase retrieval algorithm base on:
[(1) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
Phase Retrieval for High-Numerical-Aperture Optical Systems.
Optics Letters 2003, 28 (10), 801.](dx.doi.org/10.1364/OL.28.000801)

Copyright (c) 2016, David Hoffman
"""
import copy
import numpy as np
from ..phaseretrieval import *


def rp(*args, **kwargs):
    pr_result = phaseretrieval(*args, **kwargs)
    return pr_result.phase
