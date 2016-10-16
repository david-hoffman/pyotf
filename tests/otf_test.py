import numpy as np
from nose.tools import *
import unittest
from pyOTF.otf import *


class BasePSFCase(object):
    """A parent class to take care of all psf base class testing

    There's no Test in the name because I don't want it picked up by testing"""
    def test_dtypes(self):
        """Make sure dtypes make sense"""
        model = self.model
        assert np.issubdtype(model.PSFi.dtype, float), "PSFi should be a float but is a {}".format(model.PSFi.dtype)
        assert np.issubdtype(model.PSFa.dtype, complex), "PSFa should be complex but is a {}".format(model.PSFa.dtype)
        assert np.issubdtype(model.OTFi.dtype, complex), "OTFi should be complex but is a {}".format(model.OTFi.dtype)
        assert np.issubdtype(model.OTFa.dtype, complex), "OTFa should be complex but is a {}".format(model.OTFa.dtype)

    def test_PSFi_positive(self):
        """The intensity PSF should always be positive"""
        assert_true((self.model.PSFi >= 0).all())


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
        assert_tuple_equal(model.PSFi.shape, (3, 128, 128))
        # make sure changing sizes is reflected in result
        model.size = 256
        model.zrange = 0
        assert_tuple_equal(model.PSFi.shape, (1, 256, 256))


class TestSheppardPSF(unittest.TestCase, BasePSFCase):
    """Test sheppard PSF"""

    def setUp(self):
        self.model = SheppardPSF(500, 0.85, 1.0, 140, 64)

    def test_zres(self):
        """Make sure zres is set properly"""
        self.model.zres = 100
