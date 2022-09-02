#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for the pyotf module.

Copyright (c) 2016, David Hoffman
"""

import numpy as np
from dphtools.utils import fft_pad, slice_maker
from numpy.fft import fftn, fftshift, ifftn, ifftshift


def easy_fft(data, axes=None):
    """FFT that includes shifting."""
    return fftshift(fftn(ifftshift(data, axes=axes), axes=axes), axes=axes)


def easy_ifft(data, axes=None):
    """Inverse FFT that includes shifting."""
    return ifftshift(ifftn(fftshift(data, axes=axes), axes=axes), axes=axes)


def cart2pol(y, x):
    """Convert from cartesian to polar coordinates."""
    theta = np.arctan2(y, x)
    rho = np.hypot(y, x)
    return rho, theta


class NumericProperty(property):
    """Define a property that must be numeric."""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, attr=None, vartype=None):
        """Property that must be numeric.

        Parameters
        ----------
        fget : callable or None
            Callable function to get the parameter
            Must have signature fget(self)
        fset : callable or None
            Callable function to set the parameter
            Must have signature fset(self, value)
        fdel : callable or None
            Callable function to delete the parameter
            Must have signature fdel(self)
        doc : str
            Docstring for the parameter
        attr : str
            The name of the backing attribute.
        vartype : type
            The type to validate, defaults to `int`
        """
        if attr is not None and vartype is not None:
            self.attr = attr
            self.vartype = vartype

            def fget(obj):
                return getattr(obj, self.attr)

            def fset(obj, value):
                if not isinstance(value, self.vartype):
                    raise TypeError(f"{self.attr} must be an {self.vartype}, var = {value}")
                if value < 0:
                    raise ValueError(f"{self.attr} must be larger than 0")
                if getattr(obj, self.attr, None) != value:
                    setattr(obj, self.attr, value)
                    # call update code
                    obj._attribute_changed()

        super().__init__(fget, fset, fdel, doc)


def center_data(data):
    """Center data on its maximum.

    Parameters
    ----------
    data : ndarray
        Array of data points

    Returns
    -------
    centered_data : ndarray same shape as data
        data with max value at the central location of the array

    Example
    -------
    >>> data = np.array(
    ...     (
    ...         (1, 0, 0),
    ...         (0, 0, 0),
    ...         (0, 0, 0),
    ...     )
    ... )
    >>> center_data(data)
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])


    >>> data = np.array(
    ...     (
    ...         (2, 1, 0, 0),
    ...         (1, 0, 0, 0),
    ...         (0, 0, 0, 1),
    ...     )
    ... )
    >>> center_data(data)
    array([[0, 1, 0, 0],
           [0, 0, 2, 1],
           [0, 0, 1, 0]])
    """
    # copy data
    centered_data = data.copy()
    # extract shape and max location
    data_shape = data.shape
    max_loc = np.unravel_index(data.argmax(), data_shape)
    # iterate through dimensions and roll data to the right place
    for i, (x0, nx) in enumerate(zip(max_loc, data_shape)):
        centered_data = np.roll(centered_data, nx // 2 - x0, i)
    return centered_data


def remove_bg(data: np.ndarray, multiplier: float) -> np.ndarray:
    """Remove background from data.

    Utility that measures mode of data and subtracts a multiplier of it

    Example
    -------
    >>> data = np.array([1, 1, 1, 0, 2])
    >>> remove_bg(data, 1)
    array([ 0.,  0.,  0., -1.,  1.])
    """
    # TODO: should add bit for floats, that will find the mode using the hist
    # function bincounts with num bins specified
    if np.issubdtype(data.dtype, np.integer):
        if np.any(data < 0):
            raise ValueError("Negative values unsupported")
        else:
            mode = np.bincount(data.ravel()).argmax()
    else:
        raise ValueError("Floating point numbers unsupported")
    return data - multiplier * float(mode)


def psqrt(data):
    """Take the positive square root, negative values will be set to zero.

    Example
    -------
    >>> psqrt((-4, 4))
    array([0., 2.])
    """
    return np.sqrt(np.fmax(0, data))


def prep_data_for_PR(data, xysize=None, multiplier=1.05):
    """Prepare data for phase retrieval.

    Will pad or crop to xysize and remove mode times multiplier and clip at zero

    Parameters
    ----------
    data : ndarray
        The PSF data to prepare for phase retrieval
    xysize : int
        Size to pad or crop `data` to along the y, x dimensions
    multiplier : float
        The amount to by which to multiply the mode before subtracting

    Returns
    -------
    prepped_data : ndarray
        The data that has been prepped for phase retrieval.
    """
    # pull shape
    nz, ny, nx = data.shape

    # remove background
    data_without_bg = remove_bg(data, multiplier)

    # figure out padding or cropping
    if xysize is None:
        xysize = max(ny, nx)

    if xysize == ny == nx:
        pad_data = data_without_bg
    elif xysize >= max(ny, nx):
        # pad data out to the proper size, pad with zeros
        pad_data = fft_pad(data_without_bg, (nz, xysize, xysize), mode="constant")
    else:
        # if need to crop, crop and center and return
        my_slice = slice_maker(((ny + 1) // 2, (nx + 1) // 2), xysize)
        pad_data = center_data(data_without_bg)[(Ellipsis,) + my_slice]

    # return centered data
    return np.fmax(0, pad_data)
