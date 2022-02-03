#!/usr/bin/env python
# -*- coding: utf-8 -*-
# zernike.py
"""
A module defining the zernike polynomials and associated functions.

Running this file as a script will output a graph of the first 15 zernike
polynomials on the unit disk.

https://en.wikipedia.org/wiki/Zernike_polynomials
http://mathworld.wolfram.com/ZernikePolynomial.html

Copyright (c) 2016, David Hoffman
"""

import numpy as np
from scipy.special import eval_jacobi
from .utils import cart2pol

from typing import Tuple


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
    22: "secondary spherical",
}

name2noll = {v: k for k, v in noll2name.items()}


def _ingest_degrees(n, m):
    n, m = np.asarray(n), np.asarray(m)

    # check inputs
    if np.any((n < abs(m)) | ((n - m) % 2 == 1)):
        raise ValueError(f"Invalid combination of ({n}, {m}) in Noll indexing")

    if not np.issubdtype(n.dtype, np.signedinteger):
        raise ValueError("Radial degree is not integer, input = {n}")

    if not np.issubdtype(m.dtype, np.signedinteger):
        raise ValueError("Azimuthal degree is not integer, input = {m}")

    return n, m


def degrees2osa(n: int, m: int) -> int:
    """Convert the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ) to the sequential OSA/ANSI stardard index number.
    Returns NaN for invalid integer pairs.
    See also: [`mn2Noll`], [`OSA2mn`]
    Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002
    """
    n, m = _ingest_degrees(n, m)

    return ((1 / 2) * (n * (n + 2) + m)).astype(int)


def degrees2noll(n: int, m: int) -> int:
    """Convert the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ) to the sequential Noll index number.
    Returns NaN for invalid integer pairs.
    See also: [`mn2OSA`], [`Noll2mn`]
    Source: (https://en.wikipedia.org/wiki/Zernike_polynomials)
    """

    n, m = _ingest_degrees(n, m)

    p = np.full_like(m, -1)
    n_mod_4 = n % 4
    p[(m > 0) & (n_mod_4 < 2)] = 0
    p[(m < 0) & (n_mod_4 > 1)] = 0
    p[(m >= 0) & (n_mod_4 > 1)] = 1
    p[(m <= 0) & (n_mod_4 < 2)] = 1

    if np.any(p < 0):
        raise RuntimeError
    return (n * (n + 1) / 2 + abs(m) + p).astype(int)


def _ingest_index(j):
    j = np.asarray(j)

    # check inputs
    if not (j > 0).all():
        raise ValueError("Invalid Noll index")
    if not np.issubdtype(j.dtype, np.signedinteger):
        raise ValueError("Index is not integer, input = {j}")
    return j


def osa2degrees(j: int) -> Tuple[int, int]:
    """Convert the sequential OSA/ANSI stardard index number j to the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ).
    See also: [`Noll2mn`], [`mn2OSA`], [`OSA2Noll`]
    Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002
    """
    j = _ingest_index(j)

    n = (np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2)).astype(int)
    m = 2 * j - n * (n + 2)
    return n.astype(int), m.astype(int)


def noll2degrees(j: int) -> Tuple[int, int]:
    """
    Convert the Noll index number j to the integer pair (n,m) that defines the Zernike polynomial Z_n^m(ρ,θ).
    See also: [`OSA2mn`], [`mn2Noll`], [`Noll2OSA`]
    Source: (https://en.wikipedia.org/wiki/Zernike_polynomials)
    """
    j = _ingest_index(j)

    n = (np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2)).astype(int)
    jr = j - (n * (n + 1) / 2).astype(int)

    # if (n % 4) < 2:
    #     m1 = jr
    #     m2 = -(jr - 1)
    #     if (n - m1) % 2 == 0:
    #         m = m1
    #     else:
    #         m = m2
    # else:  # mod(n,4) ∈ (2,3)
    #     m1 = jr - 1
    #     m2 = -(jr)
    #     if (n - m1) % 2 == 0:
    #         m = m1
    #     else:
    #         m = m2

    # below is the vectorization version of the above.

    m1 = np.zeros_like(jr)
    m2 = np.zeros_like(jr)
    m = np.zeros_like(jr)

    n_mod_4 = n % 4

    idx0 = n_mod_4 < 2
    m1[idx0] = jr[idx0]
    m2[idx0] = -(jr[idx0] - 1)

    m1[~idx0] = jr[~idx0] - 1
    m2[~idx0] = -jr[~idx0]

    idx1 = (n - m1) % 2 == 0
    m[idx1] = m1[idx1]
    m[~idx1] = m2[~idx1]

    return n, m


# def degrees2noll(n, m):
#     """Convert from radial and azimuthal degrees to Noll's index."""
#     n, m = np.asarray(n), np.asarray(m)
#     # check inputs
#     if not np.issubdtype(n.dtype, np.signedinteger):
#         raise ValueError("Radial degree is not integer, input = {n}")
#     if not np.issubdtype(m.dtype, np.signedinteger):
#         raise ValueError("Azimuthal degree is not integer, input = {m}")
#     if ((n - m) % 2).any():
#         raise ValueError("The difference between radial and azimuthal degree isn't mod 2")
#     # do the mapping
#     mod = n % 4
#     if mod < 2:
#     return noll


def zernike(r: float, theta: float, n: int, m: int, norm: bool = True) -> float:
    """Calculate the Zernike polynomial on the unit disk for the requested orders.

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
    >>> zern = zernike(r, theta, 4, 0)  # generates the defocus zernike polynomial
    """
    n, m = np.asarray(n), np.asarray(m)
    if n.ndim > 1:
        raise ValueError("Radial degree has the wrong shape")
    if m.ndim > 1:
        raise ValueError("Azimuthal degree has the wrong shape")
    if n.shape != m.shape:
        raise ValueError("Radial and Azimuthal degrees have different shapes")

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
    return np.array([_zernike(r, theta, nn, mm, norm) for nn, mm in zip(n, m)]).squeeze()


def _radial_zernike(r, n, m):
    """Radial part of the zernike polynomial.

    Formula from http://mathworld.wolfram.com/ZernikePolynomial.html
    """
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
    jacobi = eval_jacobi(coef2, m, 0, 1 - 2 * rprime**2)
    rad_zern[valid_points] = (-1) ** coef2 * rprime**m * jacobi
    return rad_zern


def _zernike(r, theta, n, m, norm=True):
    """Calculate the full zernike polynomial."""
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
        # https://www.gatinel.com/en/recherche-formation/wavefront-sensing/zernike-polynomials/
        if m == 0:
            # m is zero
            norm = np.sqrt(n + 1)
        else:
            # m not zero
            norm = np.sqrt(2 * (n + 1))
        zern *= norm
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
