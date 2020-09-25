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
import logging

import numpy as np
from scipy.signal import fftconvolve

from .otf import HanserPSF, SheppardPSF
from .utils import (
    NumericProperty,
    bin_ndarray,
    cached_property,
    easy_fft,
    easy_ifft,
    radial_profile,
)

logger = logging.getLogger(__name__)

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

    def __init__(self, *, model, na, ni, wl, size, pixel_size, oversample_factor, **kwargs):
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
        self.pixel_size = pixel_size
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
        if self.oversample_factor > 1:
            if self.psf_params["size"] % 2 == 0:
                # if we're even then we'll be in the upper left hand corner of the super pixel
                # and we'll need to shift to the bottom and right by oversample_factor // 2
                shift = self.oversample_factor // 2
                psf = np.roll(psf, (shift, shift), axis=(1, 2))

            # only bin in the lateral direction
            psf = bin_ndarray(psf, bin_size=(1, self.oversample_factor, self.oversample_factor))

        # normalize psf
        return psf / psf.sum()

    @cached_property
    def OTF(self):
        return easy_fft(self.PSF)


def _disk_kernel(radius):
    """Model of the pinhole transmission function"""
    full_size = int(np.ceil(radius * 2))
    if full_size % 2 == 0:
        full_size += 1
    coords = np.indices((full_size, full_size)) - (full_size - 1) // 2
    r = np.sqrt((coords ** 2).sum(0))
    kernel = r < radius
    return kernel / kernel.sum()


class ConfocalMicroscope(WidefieldMicroscope):
    """A class representing a confocal microscope"""

    pinhole_size = NumericProperty(
        attr="_pinhole_size",
        vartype=(float, int),
        doc="Size of the pinhole (in airy units relative to emission wavelength",
    )

    wl_exc = NumericProperty(attr="_wl_exc", vartype=float, doc="The excitation wavelength")

    def __init__(self, *, wl_exc, pinhole_size, **kwargs):
        super().__init__(**kwargs)
        self.pinhole_size = pinhole_size

        # make the emission PSF
        self.model_exc = copy.deepcopy(self.model)
        self.model_exc.wl = wl_exc

    @property
    def model_psf(self):
        """The oversampled confocal PSF"""
        # Calculate the AU in pixels
        airy_unit = 1.22 * self.model.wl / self.model.na / self.model.res
        logger.info(f"Airy unit = {airy_unit:}")
        # Calculate the pinhole radius in pixels
        pixel_pinhole_radius = self.pinhole_size * airy_unit / 2
        #
        if pixel_pinhole_radius > 1.5:
            kernel = _disk_kernel(pixel_pinhole_radius)
            psf_det_au = fftconvolve(self.model.PSFi, kernel[None], "same", axes=(1, 2))
        else:
            psf_det_au = self.model.PSFi
        psf_con_au = psf_det_au * self.model_exc.PSFi
        return psf_con_au


class ApotomeMicroscope(WidefieldMicroscope):
    """A class representing the approximate PSF/OTF for the apotome microscope

    This is a poor approximation (see notebooks) and thus has limited functionality.

    The grid pattern is set at half NA
    
    https://www.zeiss.com/microscopy/us/products/imaging-systems/apotome-2-for-biology.html
    https://www.osapublishing.org/abstract.cfm?URI=ol-22-24-1905
    http://www.sciencedirect.com/science/article/pii/S0030401898002107
    """

    @property
    def model_psf(self):
        """The oversampled apotome PSF"""
        # make the hybrid OTF
        hybrid_otf = easy_fft(self.model.PSFi, axes=(1, 2))
        # get the radial average
        rotf = np.array([radial_profile(o)[0] for o in hybrid_otf])
        rotf /= rotf.max()
        rotf = abs(rotf)

        # define the Abbe diffraction limit in frequency space pixels.
        nyquist_sampling = self.psf_params["wl"] / self.psf_params["na"] / 4
        abbe_limit = int(
            np.rint(self.psf_params["size"] * self.psf_params["res"] / nyquist_sampling / 2)
        )

        # define the approximate axial response of the system
        axial_profile = rotf[:, abbe_limit // 2]

        psf_apotome = axial_profile[:, None, None] * self.model.PSFi
        return psf_apotome


class BaseSIMMicroscope(WidefieldMicroscope):
    """A base class for SIM and SIM like microscopes"""

    na_exc = NumericProperty(attr="_na_exc", vartype=float, doc="The excitation NA")
    wl_exc = NumericProperty(attr="_wl_exc", vartype=float, doc="The excitation wavelength")
    coherent = NumericProperty(
        attr="_coherent", vartype=bool, doc="Treat the orientations coherently?"
    )
    dc = NumericProperty(attr="_dc", vartype=bool, doc="Include the DC component")
    dc_suppress = NumericProperty(
        attr="_dc_suppress", vartype=bool, doc="Suppress the DC component"
    )

    def __init__(
        self, *, na_exc, wl_exc, wiener, coherent, dc, dc_suppress, orientations, **kwargs
    ):
        """orientations : sequence
            The different orentation angles of the excitation
        wiener : float or None
            The value of the wiener paramter, if None is passed then no deconvolution is performed
        """
        super().__init__(**kwargs)

        if na_exc is None:
            na_exc = self.psf_params["na"]
        self.na_exc = na_exc

        if wl_exc is None:
            wl_exc = self.psf_params["wl"]
        self.wl_exc = wl_exc

        self.coherent = coherent
        self.dc = dc
        self.dc_suppress = dc_suppress

        self.orientations = orientations

        self.wiener = wiener
        if self.wiener < 0:
            raise ValueError(f"self.wiener is {self.wiener:} which should be greater than zero")

    @property
    def model_psf(self):
        """The oversampled SIM PSF

        This is by no means the most general implementation, but it seems to serve
        all my use cases
        """
        # centered coordinate system.
        x = (
            np.arange(self.psf_params["size"]) - (self.psf_params["size"] + 1) // 2
        ) * self.psf_params["res"]

        z = (
            np.arange(self.psf_params["zsize"]) - (self.psf_params["zsize"] + 1) // 2
        ) * self.psf_params["zres"]

        # open grid
        zz, yy, xx = z[:, None, None], x[None, :, None], x[None, None, :]

        # magnitude of the excitation k vector (spatial frequency of a plane wave with
        # wavelength self.exc_wl)
        freq = 2 * np.pi * self.psf_params["ni"] / self.wl_exc

        # the angle in frequency space that the exciation waves make with the kz axis
        alpha = np.arcsin(self.na_exc / self.psf_params["ni"])

        # are we applying a wiener filter?
        if self.wiener is None:
            psf = self.model.PSFi
        elif self.wiener >= 0:
            # https://en.wikipedia.org/wiki/Wiener_deconvolution
            otf = abs(self.model.OTFi) ** 2
            if self.wiener == 0:
                # everything within OTF support is 1
                wiener_otf = otf > 1e-16
            else:
                w = otf.max() * self.wiener ** 2
                wiener_otf = otf / (otf + w)
            # The PSF is real, discard the imaginary part
            # we don't take the absolute value because
            # wiener deconvolution doesn't prescribe it
            psf = easy_ifft(wiener_otf).real
        else:
            raise ValueError(f"self.wiener is {self.wiener:} which should be greater than zero")

        # Are we including a DC beam?
        if self.dc:
            base_pattern = np.exp(1j * zz * freq) * np.ones(psf.shape[1:], dtype=complex)[None]
        else:
            base_pattern = np.zeros(psf.shape, dtype=complex)

        # If we're not coherent, initialize the PSF
        if not self.coherent:
            sim_psf = np.zeros_like(psf)

        # loop through orientations
        for orientation in self.orientations:
            # if we're not coherent reinitialize for the new orientation.
            if not self.coherent:
                exc_pattern = base_pattern.copy()
            else:
                exc_pattern = base_pattern

            # Calculate lateral rotated coordinate system
            rr = xx * np.cos(orientation) + yy * np.sin(orientation)

            # Again, not the most general, but covers most cases
            # There's symmetry to be exploited here for computational
            # gains.

            # build up the excitation pattern
            for theta in (-alpha, alpha):
                exc_pattern += np.exp(1j * ((rr * np.sin(theta) + zz * np.cos(theta)) * freq))

            # if we're not coherent sum the effective PSFs for each orientation
            if not self.coherent:
                sim_psf += psf * (abs(exc_pattern) ** 2)

        # if we are coherent then just calculate the total PSF
        if self.coherent:
            sim_psf = psf * (abs(exc_pattern) ** 2)
        else:
            # If we want to suppress the DC component do it here.
            if self.dc_suppress:
                sim_psf -= (2 * len(self.orientations) - 1) * psf

        return sim_psf


class SIM2DMicroscope(BaseSIMMicroscope):
    """A class for 2D-SIM, including optical sectioning SIM"""

    def __init__(self, **kwargs):
        super().__init__(coherent=False, dc=False, **kwargs)


class SIM3DMicroscope(BaseSIMMicroscope):
    """A class for 3D-SIM using multiple pattern orientations"""

    def __init__(self, **kwargs):
        super().__init__(coherent=False, dc=True, **kwargs)


class LatticeSIMMicroscope(BaseSIMMicroscope):
    """A class for Zeiss Lattice SIM

    https://www.zeiss.com/microscopy/us/products/elyra-7-with-lattice-sim-for-fast-and-gentle-3d-superresolution-microscopy.html
    """

    def __init__(self, **kwargs):
        super().__init__(coherent=True, dc=True, orientations=(0, np.pi / 2), **kwargs)


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    base_psf_params = {
        "model": "sheppard",
        "oversample_factor": 1,
        "pixel_size": 0.05,
        "na": 1.27,
        "ni": 1.33,
        "wl": 0.585,
        "size": 128,
        "vec_corr": "none",
    }

    sim_psf_params = {"na_exc": None, "wl_exc": 0.561, "wiener": 0.1, "dc_suppress": True}

    sim_psf_params.update(base_psf_params)

    orientations = (0, 2 * np.pi / 3, 4 * np.pi / 3)

    psfs = (
        WidefieldMicroscope(**base_psf_params),
        ConfocalMicroscope(**base_psf_params, pinhole_size=1.5, wl_exc=0.561),
        ConfocalMicroscope(**base_psf_params, pinhole_size=0, wl_exc=0.561),
        SIM2DMicroscope(
            orientations=orientations, **{**sim_psf_params, "na_exc": sim_psf_params["na"] / 2}
        ),
        SIM2DMicroscope(orientations=orientations, **sim_psf_params),
        SIM3DMicroscope(orientations=orientations, **sim_psf_params),
        LatticeSIMMicroscope(**sim_psf_params),
    )

    labels = ("Epi", "Confocal 1.5 AU", "AiryScan", "OS-SIM", "2D-SIM", "3D-SIM", "Lattice SIM")

    ncols = len(psfs)
    gam = 0.5
    interpolation = "bicubic"
    vmin = 1e-3
    res = base_psf_params["pixel_size"]

    assert ncols == len(labels), "Lengths mismatched"
    assert ncols < 10

    plot_size = 1.5

    fig = plt.figure(
        None,
        (plot_size * ncols, plot_size * 4),
        subplotpars=mpl.figure.SubplotParams(bottom=0.015, left=0.025, right=0.975, top=0.965,),
    )
    grid = ImageGrid(fig, 111, nrows_ncols=(4, ncols), axes_pad=0.1)

    fig2, axp = plt.subplots(
        dpi=150,
        figsize=(plot_size * ncols, 4),
        subplotpars=mpl.figure.SubplotParams(bottom=0.1, left=0.025, right=0.975, top=0.925,),
    )

    plt.set_cmap("inferno")

    for (i, p), l, col in zip(enumerate(psfs), labels, grid.axes_column):
        p = p.PSF
        p /= p.max()

        psf_plot = dict(
            norm=mpl.colors.PowerNorm(gam), interpolation=interpolation, cmap="Greys_r"
        )

        col[0].imshow(p.max(1), **psf_plot)
        col[1].imshow(p.max(0), **psf_plot)

        col[0].set_title(l)

        otf = abs(easy_fft(p))
        otf /= otf.max()
        otf = np.fmax(otf, vmin)
        c = (len(otf) + 1) // 2

        col[2].matshow(otf[:, c], norm=mpl.colors.LogNorm(), interpolation=interpolation)
        col[3].matshow(otf[c], norm=mpl.colors.LogNorm(), interpolation=interpolation)

        pp = p.sum((1, 2))
        axp.plot((np.arange(len(pp)) - (len(pp) + 1) // 2) * res, pp / pp.max(), label=l)

    for ax in grid:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    ylabels = "XZ", "XY"
    ylabels += tuple(map(lambda x: r"$k_{{{}}}$".format(x), ylabels))
    for ax, l in zip(grid.axes_column[0], ylabels):
        ax.set_ylabel(l)

    axp.yaxis.set_major_locator(plt.NullLocator())
    axp.set_xlabel("Axial Position (µm)")
    axp.set_title("On Axis Intensity")
    axp.legend()

    plt.show()
