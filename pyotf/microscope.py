#!/usr/bin/env python
# -*- coding: utf-8 -*-
# microscope.py
"""
A module to simulate optical transfer functions and point spread functions
for specific types of microscopes

Currently the available microscopes are:
- Widefield Epi
- Confocal
- Apotome

See notebooks/Microscope Imaging Models for more details

Copyright (c) 2020, David Hoffman
"""

from collections import namedtuple

from .utils import cached_property, easy_fft

MODELS = {
    "hanser": HanserPSF,
    "sheppard": SheppardPSF,
}


def choose_model(model):
    try:
        return MODELS[model.lower()]
    except KeyError:
        raise ValueError(
            f"Model {model:} doesn't exist please choose one of: " + ", ".join(MODELS.keys())
        )


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
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))

    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]

    flattened = [l for p in compression_pairs for l in p]

    ndarray = ndarray.reshape(flattened)

    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


class WidefieldMicroscope(object):
    """A base class for microscope models"""

    def __init__(self, model, na, ni, wl, size, pixel_size, oversample_factor, **kwargs):
        # set zsize here because we don't want to oversample there.
        self.psf_params = dict(na=na, ni=ni, wl=wl, zsize=size)
        self.psf_params.update(kwargs)

        assert isinstance(oversample_factor, int), "oversample_factor must be integer"
        assert oversample_factor % 2 == 1, "oversample_factor must be odd"

        if oversample_factor == 1:
            self.psf_params["size"] = size
            self.psf_params["res"] = pixel_size
        elif oversample_factor > 1:
            self.psf_params["size"] = size * oversample_factor
            self.psf_params["res"] = pixel_size / oversample_factor
        else:
            raise ValueError("oversample_factor must be positive")

        self.oversample_factor = oversample_factor

        self.model = choose_model(model)(**self.psf_params)

    def _attribute_changed(self):
        """What to do if an attribute has changed."""
        # try removing the PSF
        try:
            del self.PSF
        except AttributeError:
            pass
        # try removing the PSF
        try:
            del self.OTF
        except AttributeError:
            pass

    @cached_property
    def PSF(self):
        """The point spread function of the microscope"""
        # only bin in the lateral direction
        return bin_ndarray(
            self.model.PSFi, bin_size=(1, self.oversample_factor, self.oversample_factor)
        )

    @cached_property
    def OTF(self):
        return easy_fft(self.PSF)


class ConfocalMicroscope(WidefieldMicroscope):
    """A base class for microscope models"""

    pass


class ApotomeMicroscope(WidefieldMicroscope):
    """A base class for microscope models"""

    pass


# convenience functions


def widefield():
    pass


def confocal():
    pass


def apotome():
    pass
