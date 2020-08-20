#!/usr/bin/env python
# -*- coding: utf-8 -*-
# zernike.py
"""
A module defining the zernike polynomials and associated functions to convert
between radial and azimuthal degree pairs and Noll's indices.

Running this file as a script will output a graph of the first 15 zernike
polynomials on the unit disk.

https://en.wikipedia.org/wiki/Zernike_polynomials
http://mathworld.wolfram.com/ZernikePolynomial.html

Copyright (c) 2016, David Hoffman
"""

import numpy as np
from scipy.special import eval_jacobi
from .utils import cart2pol

# forward mapping of Noll indices https://oeis.org/A176988
noll_mapping = np.array(
    [
        1,
        3,
        2,
        5,
        4,
        6,
        9,
        7,
        8,
        10,
        15,
        13,
        11,
        12,
        14,
        21,
        19,
        17,
        16,
        18,
        20,
        27,
        25,
        23,
        22,
        24,
        26,
        28,
        35,
        33,
        31,
        29,
        30,
        32,
        34,
        36,
        45,
        43,
        41,
        39,
        37,
        38,
        40,
        42,
        44,
        55,
        53,
        51,
        49,
        47,
        46,
        48,
        50,
        52,
        54,
        65,
        63,
        61,
        59,
        57,
        56,
        58,
        60,
        62,
        64,
        66,
        77,
        75,
        73,
        71,
        69,
        67,
        68,
        70,
        72,
        74,
        76,
        78,
        91,
        89,
        87,
        85,
        83,
        81,
        79,
        80,
        82,
        84,
        86,
        88,
        90,
        105,
        103,
        101,
        99,
        97,
        95,
        93,
        92,
        94,
        96,
        98,
        100,
        102,
        104,
        119,
        117,
        115,
        113,
        111,
        109,
        107,
        106,
        108,
        110,
        112,
        114,
        116,
        118,
        120,
    ]
)

# reverse mapping of noll indices
noll_inverse = noll_mapping.argsort()

# classical names for the Noll indices
# https://en.wikipedia.org/wiki/Zernike_polynomials
noll2name = {
    1: "piston",
    2: "tip",
    3: "tilt",
    4: "defocus",
    5: "oblique astigmatism",
    6: "vertical astigmatism",
    7: "vertical coma",
    8: "horizontal coma",
    9: "vertical trefoil",
    10: "oblique trefoil",
    11: "primary spherical",
    12: "vertical secondary astigmatism",
    13: "oblique secondary astigmatism",
    14: "vertical quadrafoil",
    15: "oblique quadrafoil",
}

name2noll = {v: k for k, v in noll2name.items()}


def noll2degrees(noll):
    """Convert from Noll's indices to radial degree and azimuthal degree"""
    noll = np.asarray(noll)
    if not np.issubdtype(noll.dtype, np.signedinteger):
        raise ValueError(f"input is not integer, input = {noll}")
    if not (noll > 0).all():
        raise ValueError(f"Noll indices must be greater than 0, input = {noll}")
    # need to subtract 1 from the Noll's indices because they start at 1.
    p = noll_inverse[noll - 1]
    n = np.ceil((-3 + np.sqrt(9 + 8 * p)) / 2)
    m = 2 * p - n * (n + 2)
    return n.astype(int), m.astype(int)


def degrees2noll(n, m):
    """Convert from radial and azimuthal degrees to Noll's index"""
    n, m = np.asarray(n), np.asarray(m)
    # check inputs
    if not np.issubdtype(n.dtype, np.signedinteger):
        raise ValueError("Radial degree is not integer, input = {n}")
    if not np.issubdtype(m.dtype, np.signedinteger):
        raise ValueError("Azimuthal degree is not integer, input = {m}")
    if ((n - m) % 2).any():
        raise ValueError("The difference between radial and azimuthal degree isn't mod 2")
    # do the mapping
    p = (m + n * (n + 2)) / 2
    noll = noll_mapping[p.astype(int)]
    return noll


def zernike(r, theta, *args, **kwargs):
    """Calculates the Zernike polynomial on the unit disk for the requested
    orders

    Parameters
    ----------
    r : ndarray
    theta : ndarray

    Args
    ----
    Noll : numeric or numeric sequence
        Noll's Indices to generate
    (n, m) : tuple of numerics or numeric sequences
        Radial and azimuthal degrees
    n : see above
    m : see above

    Kwargs
    ------
    norm : bool (default False)
        Do you want the output normed?

    Returns
    -------
    zernike : ndarray
        The zernike polynomials corresponding to Noll or (n, m) whichever are
        provided

    Example
    -------
    >>> x = np.linspace(-1, 1, 512)
    >>> xx, yy = np.meshgrid(x, x)
    >>> r, theta = cart2pol(yy, xx)
    >>> zern = zernike(r, theta, 4)  # generates the defocus zernike polynomial
    """
    if len(args) == 1:
        args = np.asarray(args[0])
        if args.ndim < 2:
            n, m = noll2degrees(args)
        elif args.ndim == 2:
            if args.shape[0] == 2:
                n, m = args
            else:
                raise RuntimeError("This shouldn't happen")
        else:
            raise ValueError(f"{args.shape} is the wrong shape")
    elif len(args) == 2:
        n, m = np.asarray(args)
        if n.ndim > 1:
            raise ValueError("Radial degree has the wrong shape")
        if m.ndim > 1:
            raise ValueError("Azimuthal degree has the wrong shape")
        if n.shape != m.shape:
            raise ValueError("Radial and Azimuthal degrees have different shapes")
    else:
        raise ValueError(f"{len(args)} is an invalid number of arguments")

    # make sure r and theta are arrays
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # make sure that r is always greater than 0
    if not (r >= 0).all():
        raise ValueError("r must always be greater or equal to 0")
    if r.ndim > 2:
        raise ValueError("Input rho and theta cannot have more than two dimensions")

    # make sure that n and m are iterable
    n, m = n.ravel(), m.ravel()

    # make sure that n is always greater or equal to m
    if not (n >= abs(m)).all():
        raise ValueError("n must always be greater or equal to m")

    # return column of zernike polynomials
    return np.array([_zernike(r, theta, nn, mm, **kwargs) for nn, mm in zip(n, m)]).squeeze()


def _radial_zernike(r, n, m):
    """The radial part of the zernike polynomial

    Formula from http://mathworld.wolfram.com/ZernikePolynomial.html"""
    rad_zern = np.zeros_like(r)
    # zernike polynomials are only valid for r <= 1
    valid_points = r <= 1.0
    if m == 0 and n == 0:
        rad_zern[valid_points] = 1
        return rad_zern
    rprime = r[valid_points]
    # for the radial part m is always positive
    m = abs(m)
    # calculate the coefs
    coef1 = (n + m) // 2
    coef2 = (n - m) // 2
    jacobi = eval_jacobi(coef2, m, 0, 1 - 2 * rprime ** 2)
    rad_zern[valid_points] = (-1) ** coef1 * rprime ** m * jacobi
    return rad_zern


def _zernike(r, theta, n, m, norm=False):
    """The actual function that calculates the full zernike polynomial"""
    # remember if m is negative
    mneg = m < 0
    # going forward m is positive (Radial zernikes are only defined for
    # positive m)
    m = abs(m)
    # if m and n aren't seperated by multiple of two then return zeros
    if (n - m) % 2:
        return np.zeros_like(r)
    zern = _radial_zernike(r, n, m)
    if mneg:
        # odd zernike
        zern *= np.sin(m * theta)
    else:
        # even zernike
        zern *= np.cos(m * theta)

    # calculate the normalization factor
    if norm:
        raise NotImplementedError
    return zern


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # make coordinates
    x = np.linspace(-1, 1, 257)
    xx, yy = np.meshgrid(x, x)  # xy indexing is default
    r, theta = cart2pol(yy, xx)
    # set up plot
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    # fill out plot
    for ax, (k, v) in zip(axs.ravel(), noll2name.items()):
        zern = zernike(r, theta, k, norm=False)
        ax.imshow(
            np.ma.array(zern, mask=r > 1),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            interpolation="bicubic",
        )
        ax.set_title(v + r", $Z_{{{}}}^{{{}}}$".format(*noll2degrees(k)))
        ax.axis("off")
    fig.tight_layout()
    plt.show()
