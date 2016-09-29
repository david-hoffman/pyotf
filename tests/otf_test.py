from nose.tools import *
import unittest
from pyOTF.otf import *


class TestHanserPSF(unittest.TestCase):
    """Test HanserPSF"""

    def setUp(self):
        self.model = HanserPSF(500, 0.85, 1.0, 140, 200)

    def test_size(self):
        '''
        Make sure when size is changed the output changes accordingly
        '''

        model = self.model

        model.size = 128
        model.zrange = [-1000, 0, 1000]
        assert_tuple_equal(model.PSFi.shape, (3, 128, 128))

        model.size = 512
        model.zrange = 0
        assert_tuple_equal(model.PSFi.shape, (1, 512, 512))


class TestSheppardPSF(unittest.TestCase):
    """Test sheppard PSF"""

    def setUp(self):
        self.model = SheppardPSF(500, 0.85, 1.0, 140, 200)

    def test_zres(self):
        """Make sure zres is set properly"""
        self.model.zres = 100
