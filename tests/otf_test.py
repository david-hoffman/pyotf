#!/usr/bin/env python
# -*- coding: utf-8 -*-
# otf_test.py
"""
Test suite for otf.py

Copyright (c) 2020, David Hoffman
"""

import unittest

import numpy as np
import pytest

from pyotf.otf import *


class BasePSFCase(object):
    """A parent class to take care of all psf base class testing

    There's no Test in the name because I don't want it picked up by testing"""

    def test_dtypes(self):
        """Make sure dtypes make sense"""
        model = self.model
        assert np.issubdtype(
            model.PSFi.dtype, np.floating
        ), f"PSFi should be a float but is a {model.PSFi.dtype}"
        assert np.issubdtype(
            model.PSFa.dtype, np.complexfloating
        ), f"PSFa should be complex but is a {model.PSFa.dtype}"
        assert np.issubdtype(
            model.OTFi.dtype, np.complexfloating
        ), f"OTFi should be complex but is a f{model.OTFi.dtype}"
        assert np.issubdtype(
            model.OTFa.dtype, np.complexfloating
        ), f"OTFa should be complex but is a {model.OTFa.dtype}"

    def test_PSFi_positive(self):
        """The intensity PSF should always be positive"""
        assert (self.model.PSFi >= 0).all()

    def test_diffraction_limit(self):
        """Should raise an error if the resolution is below nyquist for the diffraction limit"""
        with pytest.raises(ValueError):
            self.model.res = self.model.wl / self.model.na


class TestHanserPSF(unittest.TestCase, BasePSFCase):
    """Test HanserPSF"""

    def setUp(self):
        self.model = HanserPSF(525, 0.85, 1.0, 140, 64)

    def test_size(self):
        """Make sure when size is changed the output changes accordingly"""
        model = self.model
        # make one size
        model.size = 128
        model.zrange = [-1000, 0, 1000]
        assert model.PSFi.shape == (3, 128, 128)
        # make sure changing sizes is reflected in result
        model.size = 256
        model.zrange = 0
        assert model.PSFi.shape == (1, 256, 256)


class TestSheppardPSF(unittest.TestCase, BasePSFCase):
    """Test sheppard PSF"""

    def setUp(self):
        self.model = SheppardPSF(500, 0.85, 1.0, 140, 64)

    def test_zres(self):
        """Make sure zres is set properly"""
        self.model.zres = 100
