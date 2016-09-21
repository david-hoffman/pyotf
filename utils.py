# -*- coding: utf-8 -*-
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
    """utility method that includes shifting"""
    return ifftshift(
        fftn(
            fftshift(
                data, axes=axes
            ), axes=axes
        ), axes=axes)


def easy_ifft(data, axes=None):
    """utility method that includes shifting"""
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
                    raise TypeError("{} must be an {}, var = {!r}".format(self.attr, self.vartype, value))
                if value <= 0:
                    raise ValueError("{} must be larger than 0".format(self.attr))
                if getattr(obj, self.attr, None) != value:
                    setattr(obj, self.attr, value)
                    # call update code
                    obj._attribute_changed()
        super().__init__(fget, fset, fdel, doc)