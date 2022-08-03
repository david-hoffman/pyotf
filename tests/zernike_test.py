#!/usr/bin/env python
# -*- coding: utf-8 -*-
# zernike_tests.py
"""
Test suite for zernike.py.

Copyright (c) 2020, David Hoffman
"""

from multiprocessing.sharedctypes import Value

import numpy as np
import pytest

from pyotf.zernike import *


@pytest.mark.parametrize("test_input", (0, -1))
def test_noll_input(test_input):
    """Make sure an error is raised if noll isn't a positive integer."""
    with pytest.raises(ValueError):
        noll2degrees(test_input)


@pytest.mark.parametrize(
    "test_func,test_input",
    (
        # test non-integer index
        (noll2degrees, (2.5,)),
        (noll2degrees, (1.0,)),
        (osa2degrees, (2.5,)),
        (osa2degrees, (1.0,)),
        # test non-integer degrees
        (degrees2noll, (1.0, 3.0)),
        (degrees2noll, (1.5, 3.5)),
        (degrees2osa, (1.0, 3.0)),
        (degrees2osa, (1.5, 3.5)),
        # test degrees not separated by 2
        (degrees2noll, (1, 2)),
        (degrees2osa, (1, 2)),
    ),
)
def test_integer_input(test_func, test_input):
    """Make sure degrees2noll and noll2degrees only accept integer inputs."""
    with pytest.raises(ValueError):
        test_func(*test_input)


@pytest.mark.parametrize(
    "forward,inverse",
    ((noll2degrees, degrees2noll), (osa2degrees, degrees2osa)),
)
def test_indices(forward, inverse):
    """Make sure that noll2degrees and degrees2noll are opposites of each other."""
    test_noll = np.random.randint(1, 36, 10)
    test_n, test_m = forward(test_noll)
    test_noll2 = inverse(test_n, test_m)
    assert (test_noll == test_noll2).all(), f"{test_noll} != {test_noll2}"


def test_n_lt_m():  # noqa: D403
    """n must always be greater than or equal to m."""
    with pytest.raises(ValueError):
        zernike(0.5, 0.0, 4, 5)


def test_noll_forward_mapping():
    """Make sure that the mapping from degrees to Noll's indices is correct."""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(
        ((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3))
    )
    j = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    n, m = degrees.T
    j_test = degrees2noll(n, m)
    assert (j_test == j).all(), f"{j_test} != {j}"


def test_noll_reverse_mapping():
    """Make sure that the mapping from Noll's indices to degrees is correct."""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(
        ((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3))
    )
    j = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    n, m = degrees.T
    n_test, m_test = noll2degrees(j)
    assert (m_test == m).all(), f"{m_test} != {m}"
    assert (n_test == n).all(), f"{n_test} != {n}"


def test_osa_forward_mapping():
    """Make sure that the mapping from degrees to Noll's indices is correct."""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(
        ((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3))
    )
    j = np.array((0, 2, 1, 4, 3, 5, 7, 8, 6, 9))
    n, m = degrees.T
    j_test = degrees2osa(n, m)
    assert (j_test == j).all(), f"{j_test} != {j}"


def test_fringe_forward_mapping():
    """Make sure that the mapping from degrees to Noll's indices is correct."""
    # from https://en.wikipedia.org/wiki/Zernike_polynomials
    degrees = np.array(
        ((0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (3, -1), (3, 1), (3, -3), (3, 3))
    )
    j = np.array((1, 2, 3, 4, 6, 5, 8, 7, 11, 10))
    n, m = degrees.T
    j_test = degrees2fringe(n, m)
    assert (j_test == j).all(), f"{j_test} != {j}"


def test_r_theta_dims():
    """Make sure that a ValueError is raised if the dims are greater than 2."""
    r = np.ones((3, 3, 3))
    with pytest.raises(ValueError):
        zernike(r, r, 0, 0)


def test_zernike_return_shape():
    """Make sure that the return shape matches input shape."""
    x = np.linspace(-1, 1, 512)
    xx, yy = np.meshgrid(x, x)
    r, theta = cart2pol(yy, xx)
    zern = zernike(r, theta, 0, 0)
    assert zern.shape == r.shape


@pytest.mark.parametrize(
    "test_input",
    (
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
    """Make sure same result is obtained for integer and float."""
    n, m = choose_random_nm()
    r = 0.5
    theta = np.random.rand() * 2 * np.pi - np.pi
    assert np.isfinite(zernike(r, theta, n, m)).all(), f"r, theta, n, m = {r}, {theta}, {n}, {m}"


@pytest.mark.parametrize("num", (0, 1))
def test_zernike_edges(num):
    """Make sure same result is obtained at 0 and 0.0 and 1 and 1.0."""
    n, m = choose_random_nm()
    theta = np.random.rand() * 2 * np.pi - np.pi
    assert zernike(float(num), theta, n, m) == zernike(
        int(num), theta, n, m
    ), f"theta, n, m = {theta}, {n}, {m}"


def test_odd_nm():
    """Make sure that n and m seperated by odd numbers gives zeros."""
    n, m = choose_random_nm(True)
    theta = np.random.rand(100) * 2 * np.pi - np.pi
    # we'll check outside the normal range too, when r
    r = np.random.rand(100) * 2
    assert (zernike(r, theta, n, m) == 0).all(), f"theta, n, m = {theta}, {n}, {m}"


def choose_random_nm(odd=False):
    """Choose random n and m.

    Optional argument specifies whether n and m are seperated by a factor of 2 or not.
    """
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


def test_norm():
    """Test that normalization works."""
    # set up coordinates
    x = np.linspace(-1, 1, 2048)
    xx, yy = np.meshgrid(x, x)  # xy indexing is default
    r, theta = cart2pol(yy, xx)
    # fill out plot
    for (n, m), v in sorted(degrees2name.items())[0:]:
        zern = zernike(r, theta, n, m, norm=True)
        tol = 10.0 ** (n - 6)
        np.testing.assert_allclose(
            1.0, np.sqrt((zern[r <= 1] ** 2).mean()), err_msg=f"{v} failed!", atol=tol, rtol=tol
        )


# the expected value is more difficult than this
# def test_pv():
#     """Test that normalization works."""
#     # set up coordinates
#     x = np.linspace(-1, 1, 2048)
#     xx, yy = np.meshgrid(x, x)  # xy indexing is default
#     r, theta = cart2pol(yy, xx)
#     # fill out plot
#     for (n, m), v in sorted(degrees2name.items())[1:]:
#         zern = zernike(r, theta, n, m, norm=False)
#         zern_flat = zern[r <= 1]
#         np.testing.assert_allclose(
#             2.0, zern_flat.max() - zern_flat.min(), err_msg=f"{v} failed!", atol=1e-2, rtol=1e-3
#         )


def test_norm_sum():
    """Test RMS of sum of zernikes is the square root of the sum of the coefficients."""
    # set up coordinates
    x = np.linspace(-1, 1, 2048)
    xx, yy = np.meshgrid(x, x)  # xy indexing is default
    r, theta = cart2pol(yy, xx)
    # make coefs
    np.random.seed(12345)
    c0, c1 = np.random.randn(2)
    astig_zern = c0 * zernike(r, theta, 2, -2, norm=True)
    spherical_zern = c1 * zernike(r, theta, 3, -3, norm=True)
    np.testing.assert_allclose(
        abs(c0), np.sqrt((astig_zern[r <= 1] ** 2).mean()), atol=1e-3, rtol=1e-3
    )

    np.testing.assert_allclose(
        abs(c1), np.sqrt((spherical_zern[r <= 1] ** 2).mean()), atol=1e-3, rtol=1e-3
    )

    np.testing.assert_allclose(
        np.sqrt(c0**2 + c1**2),
        np.sqrt(((astig_zern + spherical_zern)[r <= 1] ** 2).mean()),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("mapping", (noll2name, osa2name, fringe2name))
def test_mapping_monotonic(mapping):
    assert (np.diff(list(mapping.keys())) > 0).all()
