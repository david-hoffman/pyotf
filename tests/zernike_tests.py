# zernike_tests.py
"""
Test suite for zernike.py
"""

from nose.tools import *
# import warnings
# from skimage.external import tifffile as tif
from simrecon.zernike import *
# import os
import numpy as np
# import unittest


def test_degrees_input():
    """Make sure an error is returned if n and m aren't seperated by two"""
    assert_raises(ValueError, degrees2noll, 1, 2)


def test_noll_input():
    """Make sure an error is raised if noll isn't a positive integer"""
    assert_raises(ValueError, noll2degrees, 0)
    assert_raises(ValueError, noll2degrees, -1)


def test_integer_input():
    """make sure degrees2noll and noll2degrees only accept integer inputs"""
    assert_raises(ValueError, noll2degrees, 2.5)
    assert_raises(ValueError, noll2degrees, 1.0)
    assert_raises(ValueError, degrees2noll, 1.0, 3.0)
    assert_raises(ValueError, degrees2noll, 1.5, 3.5)


def test_indices():
    """Make sure that noll2degrees and degrees2noll are opposites of each
    other"""
    test_noll = np.random.randint(1, 36, 10)
    test_n, test_m = noll2degrees(test_noll)
    test_noll2 = degrees2noll(test_n, test_m)
    assert_true((test_noll == test_noll2).all(),
                "{} != {}".format(test_noll, test_noll2))


def test_forward_mapping():
    """Make sure that the mapping from degrees to Noll's indices is correct"""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2),
                        (3, -1), (3, 1), (3, -3), (3, 3)))
    j = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    n, m = degrees.T
    assert_true((degrees2noll(n, m) == j).all())


def test_reverse_mapping():
    """Make sure that the mapping from Noll's indices to degrees is correct"""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2),
                        (3, -1), (3, 1), (3, -3), (3, 3)))
    j = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    n, m = degrees.T
    n_test, m_test = noll2degrees(j)
    assert_true((m_test == m).all(), "{} != {}".format(m_test, m))
    assert_true((n_test == n).all(), "{} != {}".format(n_test, n))


def test_r_theta_dims():
    """Make sure that a ValueError is raised if the dims are greater than 2"""
    r = np.ones((3, 3, 3))
    assert_raises(ValueError, zernike, r, r, 10)


def test_zernike_return_shape():
    """Make sure that the return shape matches input shape"""
    x = np.linspace(-1, 1, 512)
    xx, yy = np.meshgrid(x, x)
    r, theta = cart2pol(yy, xx)
    zern = zernike(r, theta, 10)
    assert_equal(zern.shape, r.shape)


def test_zernike_errors():
    """Make sure zernike doesn't accept bad input."""
    noll = np.ones((2, 2, 2))
    assert_raises(ValueError, zernike, 0, 0, noll)
    assert_raises(ValueError, zernike, 0, 0, noll, noll)
