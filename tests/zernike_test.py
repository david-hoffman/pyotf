#!/usr/bin/env python
# -*- coding: utf-8 -*-
# zernike_tests.py
"""
Test suite for zernike.py

Copyright (c) 2020, David Hoffman
"""

import numpy as np
import pytest

from pyotf.zernike import *


def test_degrees_input():
    """Make sure an error is returned if n and m aren't seperated by two"""
    with pytest.raises(ValueError):
        degrees2noll(1, 2)


@pytest.mark.parametrize("test_input", (0, -1))
def test_noll_input(test_input):
    """Make sure an error is raised if noll isn't a positive integer"""
    with pytest.raises(ValueError):
        noll2degrees(test_input)


@pytest.mark.parametrize(
    "test_func,test_input",
    (
        (noll2degrees, (2.5,)),
        (noll2degrees, (1.0,)),
        (degrees2noll, (1.0, 3.0)),
        (degrees2noll, (1.5, 3.5)),
    ),
)
def test_integer_input(test_func, test_input):
    """make sure degrees2noll and noll2degrees only accept integer inputs"""
    with pytest.raises(ValueError):
        test_func(*test_input)


def test_indices():
    """Make sure that noll2degrees and degrees2noll are opposites of each
    other"""
    test_noll = np.random.randint(1, 36, 10)
    test_n, test_m = noll2degrees(test_noll)
    test_noll2 = degrees2noll(test_n, test_m)
    assert (test_noll == test_noll2).all(), f"{test_noll} != {test_noll2}"


def test_n_lt_m():
    """n must always be greater than or equal to m"""
    with pytest.raises(ValueError):
        zernike(0.5, 0.0, 4, 5)


def test_forward_mapping():
    """Make sure that the mapping from degrees to Noll's indices is correct"""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(
        ((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3))
    )
    j = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    n, m = degrees.T
    assert (degrees2noll(n, m) == j).all()


def test_reverse_mapping():
    """Make sure that the mapping from Noll's indices to degrees is correct"""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(
        ((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3))
    )
    j = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    n, m = degrees.T
    n_test, m_test = noll2degrees(j)
    assert (m_test == m).all(), f"{m_test} != {m}"
    assert (n_test == n).all(), f"{n_test} != {n}"


def test_r_theta_dims():
    """Make sure that a ValueError is raised if the dims are greater than 2"""
    r = np.ones((3, 3, 3))
    with pytest.raises(ValueError):
        zernike(r, r, 10)


def test_zernike_return_shape():
    """Make sure that the return shape matches input shape"""
    x = np.linspace(-1, 1, 512)
    xx, yy = np.meshgrid(x, x)
    r, theta = cart2pol(yy, xx)
    zern = zernike(r, theta, 10)
    assert zern.shape == r.shape


@pytest.mark.parametrize(
    "test_input",
    (
        (0, 0, np.ones((2, 2, 2))),  # check noll dims
        (
            0,
            0,
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)),
        ),  # check that n and m must have dimension of 1
        (-1, 0, 0, 1),  # check that r can't be negative
        (np.ones((10, 10, 2)), 0, 0, 1),  # check that r only has 2 dims
    ),
)
def test_zernike_errors(test_input):
    """Make sure zernike doesn't accept bad input."""
    with pytest.raises(ValueError):
        zernike(*test_input)


def test_zernike_zero():
    """Make sure same result is obtained for integer and float"""
    n, m = choose_random_nm()
    r = 0.5
    theta = np.random.rand() * 2 * np.pi - np.pi
    assert np.isfinite(zernike(r, theta, n, m)).all(), f"r, theta, n, m = {r}, {theta}, {n}, {m}"


@pytest.mark.parametrize("num", (0, 1))
def test_zernike_edges(num):
    """Make sure same result is obtained at 0 and 0.0 and 1 and 1.0"""
    n, m = choose_random_nm()
    theta = np.random.rand() * 2 * np.pi - np.pi
    assert zernike(float(num), theta, n, m) == zernike(
        int(num), theta, n, m
    ), f"theta, n, m = {theta}, {n}, {m}"


def test_odd_nm():
    """Make sure that n and m seperated by odd numbers gives zeros"""
    n, m = choose_random_nm(True)
    theta = np.random.rand(100) * 2 * np.pi - np.pi
    # we'll check outside the normal range too, when r
    r = np.random.rand(100) * 2
    assert (zernike(r, theta, n, m) == 0).all(), f"theta, n, m = {theta}, {n}, {m}"


def choose_random_nm(odd=False):
    """Small utility function to choose random n and m, optional argument specifies
    whether n and m are seperated by a factor of 2 or not"""
    m = np.nan
    n = np.nan
    # make sure m and n are seperated by a factor of 2 otherwise
    # we'll get all zeros
    while (m - n + odd) % 2:
        # choose random positive n
        n = np.random.randint(100)
        if n:
            # if n is greater than zero choose random m such that
            # n >= m
            m = np.random.randint(-n, n + 1)
        else:
            m = 0
    assert n >= abs(m), f"Somethings very wrong {n} not >= {m}"
    assert not (m - n + odd) % 2, f"m = {m}, n = {n}"
    return n, m
