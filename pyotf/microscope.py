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

import copy

import numpy as np
from scipy.signal import fftconvolve

from .otf import HanserPSF, SheppardPSF
from .utils import cached_property, easy_fft, bin_ndarray, NumericProperty

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


class WidefieldMicroscope(object):
    """A base class for microscope models"""

    oversample_factor = NumericProperty(
        attr="_oversample_factor",
        vartype=int,
        doc="By how much do you want to oversample the simulation",
    )

    def __init__(self, model, na, ni, wl, size, pixel_size, oversample_factor, **kwargs):
        # set zsize and zres here because we don't want to oversample there.

        self.oversample_factor = oversample_factor

        self.psf_params = dict(na=na, ni=ni, wl=wl, zres=pixel_size, zsize=size)
        self.psf_params.update(kwargs)

        assert self.oversample_factor % 2 == 1, "oversample_factor must be odd"

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

    @property
    def model_psf(self):
        return self.model.PSFi

    @cached_property
    def PSF(self):
        """The point spread function of the microscope"""
        psf = self.model_psf
        if self.oversample_factor == 1:
            return psf
        if self.psf_params["size"] % 2 == 0:
            # if we're even then we'll be in the upper left hand corner of the super pixel
            # and we'll need to shift to the bottom and right by oversample_factor // 2
            shift = self.oversample_factor // 2
            psf = np.roll(psf, (shift, shift), axis=(1, 2))

        # only bin in the lateral direction
        return bin_ndarray(psf, bin_size=(1, self.oversample_factor, self.oversample_factor))

    @cached_property
    def OTF(self):
        return easy_fft(self.PSF)

def _disk_kernel(radius):
    """Model of the pinhole transmission function"""
    full_size = int(np.ceil(radius * 2))
    if full_size % 2 == 0:
        full_size += 1
    coords = np.indices((full_size, full_size)) - (full_size - 1) // 2
    r = np.sqrt((coords**2).sum(0))
    kernel = r < radius
    return kernel / kernel.sum()


class ConfocalMicroscope(WidefieldMicroscope):
    """A base class for microscope models"""

    pinhole_size = NumericProperty(
        attr="_pinhole_size",
        vartype=(float, int),
        doc="Size of the pinhole (in airy units relative to emission wavelength",
    )

    def __init__(self, *args, wl_em, pinhole_size, **kwargs):
        """zrange : array-like
            An alternate way to specify the z range for the calculation
            must be expressed in the same units as wavelength
        """
        super().__init__(*args, **kwargs)
        self.pinhole_size = pinhole_size
        if self.pinhole_size < 0:
            raise ValueError("pinhole_size cannot be less than 0")

        # make the emission PSF
        self.model_em = copy.deepcopy(self.model)
        self.model_em.wl = wl_em


    @property
    def model_psf(self):
        if self.pinhole_size > 0:
            airy_unit = 1.22 * self.model.wl / self.model.na / self.model.res
            kernel = _disk_kernel(self.pinhole_size * airy_unit / 2)
            psf_det_au = fftconvolve(self.model_em.PSFi, kernel[None], "same", axes=(1,2))
        else:
            psf_det_au = self.model_em.PSFi
        psf_con_au = psf_det_au * self.model.PSFi
        return psf_con_au


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
