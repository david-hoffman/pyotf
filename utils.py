# -*- coding: utf-8 -*-
"""Utility functions for the pyOTF module"""
import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import (ifftshift, fftshift,
                                             fftn, ifftn)
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import ifftshift, fftshift, fftn, ifftn


def easy_fft(data, axes=None):
    """utility method that includes fft shifting"""
    return ifftshift(
        fftn(
            fftshift(
                data, axes=axes
            ), axes=axes
        ), axes=axes)


def easy_ifft(data, axes=None):
    """utility method that includes fft shifting"""
    return fftshift(
        ifftn(
            ifftshift(
                data, axes=axes
            ), axes=axes
        ), axes=axes)


def cart2pol(y, x):
    """utility function for converting from cartesian to polar"""
    theta = np.arctan2(y, x)
    rho = np.hypot(y, x)
    return rho, theta


class NumericProperty(property):
    """A property that must be numeric.

    Parameters
    ----------
    attr : str
        The name of the backing attribute.
    vartype : type
        The type to validate, defaults to `int`
    doc : str
        Docstring for the parameter

    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None,
                 attr=None, vartype=None):
        if attr is not None and vartype is not None:
            self.attr = attr
            self.vartype = vartype

            def fget(obj):
                return getattr(obj, self.attr)

            def fset(obj, value):
                if not isinstance(value, self.vartype):
                    raise TypeError(
                        "{} must be an {}, var = {!r}".format(
                            self.attr, self.vartype, value)
                    )
                if value <= 0:
                    raise ValueError(
                        "{} must be larger than 0".format(self.attr)
                    )
                if getattr(obj, self.attr, None) != value:
                    setattr(obj, self.attr, value)
                    # call update code
                    obj._attribute_changed()
        super().__init__(fget, fset, fdel, doc)


def center_data(data):
    """Utility to center the data

    Parameters
    ----------
    data : ndarray
        Array of data points

    Returns
    -------
    centered_data : ndarray same shape as data
        data with max value at the central location of the array
    """
    # copy data
    centered_data = data.copy()
    # extract shape and max location
    data_shape = data.shape
    max_loc = np.unravel_index(data.argmax(), data_shape)
    # iterate through dimensions and roll data to the right place
    for i, (x0, nx) in enumerate(zip(max_loc, data_shape)):
        centered_data = np.roll(centered_data, (nx + 1) // 2 - x0, i)
    return centered_data


def remove_bg(data, multiplier=1.5):
    """Utility that measures mode of data and subtracts it"""
    mode = np.bincount(data.ravel()).argmax()
    return data - multiplier * mode


def psqrt(data):
    """Take the positive square root, negative values will be set to zero."""
    sdata = np.zeros_like(data, float)
    sdata[data > 0] = np.sqrt(data[data > 0])
    return sdata
