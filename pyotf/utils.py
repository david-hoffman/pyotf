#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py
"""
Utility functions for the pyotf module
Copyright (c) 2016, David Hoffman
"""

import numpy as np
import scipy.fftpack
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.ndimage._ni_support import _normalize_sequence


def _calc_crop(s1, s2):
    """Calc the cropping from the padding"""
    a1 = abs(s1) if s1 < 0 else None
    a2 = s2 if s2 < 0 else None
    return slice(a1, a2, None)


def _calc_pad(oldnum, newnum):
    """ Calculate the proper padding for fft_pad

    We have three cases:
    old number even new number even
    >>> _calc_pad(10, 16)
    (3, 3)

    old number odd new number even
    >>> _calc_pad(11, 16)
    (3, 2)

    old number odd new number odd
    >>> _calc_pad(11, 17)
    (3, 3)

    old number even new number odd
    >>> _calc_pad(10, 17)
    (4, 3)

    same numbers
    >>> _calc_pad(17, 17)
    (0, 0)

    from larger to smaller.
    >>> _calc_pad(17, 10)
    (-4, -3)
    """
    # how much do we need to add?
    width = newnum - oldnum
    # calculate one side, smaller
    pad_s = width // 2
    # calculate the other, bigger
    pad_b = width - pad_s
    pad1, pad2 = pad_b, pad_s
    return pad1, pad2


def _padding_slices(oldshape, newshape):
    """This function takes the old shape and the new shape and calculates
    the required padding or cropping.newshape

    Can be used to generate the slices needed to undo fft_pad above"""
    # generate pad widths from new shape
    padding = tuple(
        _calc_pad(o, n) if n is not None else _calc_pad(o, o) for o, n in zip(oldshape, newshape)
    )
    # Make a crop list, if any of the padding is negative
    slices = tuple(_calc_crop(s1, s2) for s1, s2 in padding)
    # leave 0 pad width where it was cropped
    padding = [(max(s1, 0), max(s2, 0)) for s1, s2 in padding]
    return padding, slices


def fft_pad(array, newshape=None, mode="median", **kwargs):
    """Pad an array to prep it for fft"""
    # pull the old shape
    oldshape = array.shape
    if newshape is None:
        # update each dimension to a 5-smooth hamming number
        newshape = tuple(scipy.fftpack.helper.next_fast_len(n) for n in oldshape)
    else:
        if isinstance(newshape, int):
            newshape = tuple(newshape for n in oldshape)
        else:
            newshape = tuple(newshape)
    # generate padding and slices
    padding, slices = _padding_slices(oldshape, newshape)
    return np.pad(array[slices], padding, mode=mode, **kwargs)


def slice_maker(xs, ws):
    """
    A utility function to generate slices for later use.

    Parameters
    ----------
    y0 : int
        center y position of the slice
    x0 : int
        center x position of the slice
    width : int
        Width of the slice

    Returns
    -------
    slices : list
        A list of slice objects, the first one is for the y dimension and
        and the second is for the x dimension.

    Notes
    -----
    The method will automatically coerce slices into acceptable bounds.

    Examples
    --------
    >>> slice_maker((30,20),10)
    [slice(25, 35, None), slice(15, 25, None)]
    >>> slice_maker((30,20),25)
    [slice(18, 43, None), slice(8, 33, None)]
    """
    # normalize inputs
    xs = np.asarray(xs)
    ws = np.asarray(_normalize_sequence(ws, len(xs)))
    if not np.isrealobj((xs, ws)):
        raise TypeError("`slice_maker` only accepts real input")
    if np.any(ws < 0):
        raise ValueError(f"width cannot be negative, width = {ws}")
    # ensure integers
    xs = np.rint(xs).astype(int)
    ws = np.rint(ws).astype(int)
    # use _calc_pad
    toreturn = []
    for x, w in zip(xs, ws):
        half2, half1 = _calc_pad(0, w)
        xstart = x - half1
        xend = x + half2
        assert xstart <= xend, "xstart > xend"
        if xend <= 0:
            xstart, xend = 0, 0
        # the max calls are to make slice_maker play nice with edges.
        toreturn.append(slice(max(0, xstart), xend))
    # return a list of slices
    return tuple(toreturn)


def easy_fft(data, axes=None):
    """utility method that includes fft shifting"""
    return fftshift(fftn(ifftshift(data, axes=axes), axes=axes), axes=axes)


def easy_ifft(data, axes=None):
    """utility method that includes fft shifting"""
    return ifftshift(ifftn(fftshift(data, axes=axes), axes=axes), axes=axes)


def cart2pol(y, x):
    """utility function for converting from cartesian to polar"""
    theta = np.arctan2(y, x)
    rho = np.hypot(y, x)
    return rho, theta


class NumericProperty(property):
    """Define a property that must be numeric"""

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, attr=None, vartype=None):
        """A property that must be numeric.

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


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    
    This will be part of the standard library starting in 3.8
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


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
        centered_data = np.roll(centered_data, nx // 2 - x0, i)
    return centered_data


def remove_bg(data, multiplier=1.5):
    """Utility that measures mode of data and subtracts a multiplier of it"""
    # should add bit for floats, that will find the mode using the hist
    # function bincounts with num bins specified
    mode = np.bincount(data.ravel()).argmax()
    return data - multiplier * mode


def psqrt(data):
    """Take the positive square root, negative values will be set to zero."""
    return np.sqrt(np.fmax(0, data))


def prep_data_for_PR(data, xysize=None, multiplier=1.5):
    """A utility to prepare data for phase retrieval

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
        pad_data = fft_pad(data_without_bg, (nz, xysize, xysize), mode="constant")
    else:
        # if need to crop, crop and center and return
        my_slice = slice_maker(((ny + 1) // 2, (nx + 1) // 2), xysize)
        return center_data(data_without_bg)[[Ellipsis] + my_slice]
    # return centered data
    return np.fmax(0, center_data(pad_data))


def bin_ndarray(ndarray, new_shape=None, bin_size=None, operation="sum"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Parameters
    ----------
    ndarray : array like object (can be dask array)
    new_shape : iterable (optional)
        The new size to bin the data to
    bin_size : scalar or iterable (optional)
        The size of the new bins

    Returns
    -------
    binned array.
    """
    if new_shape is None:
        # if new shape isn't passed then calculate it
        if bin_size is None:
            # if bin_size isn't passed then raise error
            raise ValueError("Either new shape or bin_size must be passed")
        # pull old shape
        old_shape = np.array(ndarray.shape)
        # calculate new shape, integer division!
        new_shape = old_shape // bin_size
        # calculate the crop window
        crop = tuple(slice(None, -r) if r else slice(None) for r in old_shape % bin_size)
        # crop the input array
        ndarray = ndarray[crop]
    # proceed as before
    operation = operation.lower()
    if operation not in {"sum", "mean"}:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError(f"Shape mismatch: {ndarray.shape} -> {new_shape}")

    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]

    flattened = [l for p in compression_pairs for l in p]

    ndarray = ndarray.reshape(flattened)

    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def radial_profile(data, center=None, binsize=1.0):
    """Take the radial average of a 2D data array

    Adapted from http://stackoverflow.com/a/21242776/5030014

    Parameters
    ----------
    data : ndarray (2D)
        the 2D array for which you want to calculate the radial average
    center : sequence
        the center about which you want to calculate the radial average
    binsize : sequence
        Size of radial bins, numbers less than one have questionable utility

    Returns
    -------
    radial_mean : ndarray
        a 1D radial average of data
    radial_std : ndarray
        a 1D radial standard deviation of data

    Examples
    --------
    >>> radial_profile(np.ones((11, 11)))
    (array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))
    """
    # test if the data is complex
    if np.iscomplexobj(data):
        # if it is complex, call this function on the real and
        # imaginary parts and return the complex sum.
        real_prof, real_std = radial_profile(np.real(data), center, binsize)
        imag_prof, imag_std = radial_profile(np.imag(data), center, binsize)
        return real_prof + imag_prof * 1j, np.sqrt(real_std ** 2 + imag_std ** 2)
        # or do mag and phase
        # mag_prof, mag_std = radial_profile(np.abs(data), center, binsize)
        # phase_prof, phase_std = radial_profile(np.angle(data), center, binsize)
        # return mag_prof * np.exp(phase_prof * 1j), mag_std * np.exp(phase_std * 1j)
    # pull the data shape
    idx = np.indices((data.shape))
    if center is None:
        # find the center
        center = np.array(data.shape) // 2
    else:
        # make sure center is an array.
        center = np.asarray(center)
    # calculate the radius from center
    idx2 = idx - center[(Ellipsis,) + (np.newaxis,) * (data.ndim)]
    r = np.sqrt(np.sum([i ** 2 for i in idx2], 0))
    # convert to int
    r = np.round(r / binsize).astype(np.int)
    # sum the values at equal r
    tbin = np.bincount(r.ravel(), data.ravel())
    # sum the squares at equal r
    tbin2 = np.bincount(r.ravel(), (data ** 2).ravel())
    # find how many equal r's there are
    nr = np.bincount(r.ravel())
    # calculate the radial mean
    # NOTE: because nr could be zero (for missing bins) the results will
    # have NaN for binsize != 1
    radial_mean = tbin / nr
    # calculate the radial std
    radial_std = np.sqrt(tbin2 / nr - radial_mean ** 2)
    # return them
    return radial_mean, radial_std
