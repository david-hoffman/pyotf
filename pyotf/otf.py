#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
A module to simulate optical transfer functions and point spread functions

If this file is run as a script (python -m pyotf.otf) it will compare
the HanserPSF to the SheppardPSF in a plot.

https://en.wikipedia.org/wiki/Optical_transfer_function
https://en.wikipedia.org/wiki/Point_spread_function

Copyright (c) 2020, David Hoffman
"""

import copy
import logging
from functools import cached_property

import numpy as np
from numpy.fft import fftfreq, fftshift, ifftn
from numpy.linalg import norm

from .utils import NumericProperty, cart2pol, easy_fft, easy_ifft, psqrt
from .zernike import name2noll, zernike

logger = logging.getLogger(__name__)


class BasePSF(object):
    """A base class for objects that can calculate OTF's and PSF's.
    It is not intended to be used alone

    To fully describe a PSF or OTF of an objective lens, assuming no
    abberation, we generally need a few parameters:
    - The wavelength of operation (assume monochromatic light)
    - the numerical aperature of the objective
    - the index of refraction of the medium

    For numerical calculations we'll also want to know the x/y resolution
    and number of points. Note that it is assumed that z is the optical
    axis of the objective lens"""

    # Define all the numeric properties of the base class
    wl = NumericProperty(attr="_wl", vartype=(float, int), doc="Wavelength of emission, in nm")
    na = NumericProperty(attr="_na", vartype=(float, int), doc="Numerical Aperature")
    ni = NumericProperty(attr="_ni", vartype=(float, int), doc="Refractive index")
    size = NumericProperty(attr="_size", vartype=int, doc="x/y size")
    zsize = NumericProperty(attr="_zsize", vartype=int, doc="z size")

    def __init__(
        self, wl, na, ni, res, size, zres=None, zsize=None, vec_corr="none", condition="sine"
    ):
        """Generate a PSF object

        Parameters
        ----------
        wl : numeric
            Emission wavelength of the simulation
        na : numeric
            Numerical aperature of the simulation
        ni : numeric
            index of refraction for the media
        res : numeric
            x/y resolution of the simulation, must have same units as wl
        size : int
            x/y size of the simulation

        Optional Parameters
        -------------------
        zres : numeric
            z resolution of simuation, must have same units a wl
        zsize : int
            z size of simulation
        vec_corr : str
            keyword to indicate whether to include vectorial effects
                Valid options are: "none", "x", "y", "z", "total"
                Default is: "none"
        condition : str
            keyword to indicate whether to model the sine or herschel conditions
            **Herschel's Condition** invariance of axial magnification
            **Abbe's Sine Condition** invariance of lateral magnification
            conditions
                Valid options are: "none", "sine", "herschel"
                Default is: "sine"
                Note: "none" is not a physical solution
        """
        self.wl = wl
        self.na = na
        self.ni = ni
        self.res = res
        self.size = size
        # if zres is not passed, set it to res
        if zres is None:
            zres = res
        self.zres = zres
        # if zsize isn't passed set it to size
        if zsize is None:
            zsize = size
        self.zsize = zsize
        self.vec_corr = vec_corr
        self.condition = condition

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(wl={self.wl}, na={self.na}, ni={self.ni},"
            + f" res={self.res}, size={self.size}, zres={self.zres}, zsize={self.zsize},"
            + f" vec_corr='{self.vec_corr}', condition='{self.condition}')"
        )

    def _attribute_changed(self):
        """Called whenever key attributes are changed
        Sets internal state variables to None so that when the
        user asks for them they are recalculated"""
        for attr in ("PSFa", "OTFa", "PSFi", "OTFi"):
            try:
                delattr(self, attr)
            except AttributeError:
                logger.debug(f"{attr} wasn't available to delete")

    @property
    def zres(self):
        """z resolution (nm)"""
        return self._zres

    @zres.setter
    def zres(self, value):
        # make sure z res is positive
        if not value > 0:
            raise ValueError("zres must be positive")
        self._zres = value
        self._attribute_changed()

    @property
    def res(self):
        """x/y resolution (nm)"""
        return self._res

    @res.setter
    def res(self, value):
        # max_val is the abbe limit, but for an accurate simulation
        # the pixel size must be smaller than half this number
        # thinking in terms of the convolution that is implicitly
        # performed when generating the OTFi we also don't want
        # any wrapping effects. However, allowing the number
        # to be the Abbe limit can allow phase retrieval for
        # larger pixels
        abbe_limit = 1 / (2 * self.na / self.wl)
        if value >= abbe_limit:
            raise ValueError(
                f"{value} is larger than the Abbe Limit, try a number smaller than {abbe_limit}"
            )
        if value >= abbe_limit / 2:
            logger.warning(
                f"res has been set to {value} which is greater than the Nyquist limit of {abbe_limit / 2}"
            )
        self._res = value
        self._attribute_changed()

    @property
    def vec_corr(self):
        """Whether to apply a correction to take into account the vectorial nature of
        light. Valid values are: "none", "x", "y", "z", "total"
        """
        return self._vec_corr

    @vec_corr.setter
    def vec_corr(self, value):
        valid_values = {"none", "x", "y", "z", "total"}
        if value not in valid_values:
            raise ValueError("Vector correction must be one of {}".format(", ".join(valid_values)))
        self._vec_corr = value
        self._attribute_changed()

    @property
    def condition(self):
        """Which imaging condition to simulate?"""
        return self._condition

    @condition.setter
    def condition(self, value):
        valid_values = {"none", "sine", "herschel"}
        if value not in valid_values:
            raise ValueError(("Condition must be one of {}").format(", ".join(valid_values)))
        self._condition = value
        self._attribute_changed()

    @cached_property
    def OTFa(self):
        """Amplitude OTF (coherent transfer function), complex array"""
        raise NotImplementedError

    @cached_property
    def PSFa(self):
        """Amplitude PSF, complex array"""
        raise NotImplementedError

    @cached_property
    def PSFi(self):
        """Intensity PSF, real array"""
        return (abs(self.PSFa) ** 2).sum(axis=0)

    @cached_property
    def OTFi(self):
        """Intensity OTF, complex array"""
        return easy_fft(self.PSFi)


class HanserPSF(BasePSF):
    """A class defining the pupil function and its closely related methods.

    Based on the following work

    [(1) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
    Phase-Retrieved Pupil Functions in Wide-Field Fluorescence Microscopy.
    Journal of Microscopy 2004, 216 (1), 32–48.](dx.doi.org/10.1111/j.0022-2720.2004.01393.x)
    [(2) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
    Phase Retrieval for High-Numerical-Aperture Optical Systems.
    Optics Letters 2003, 28 (10), 801.](dx.doi.org/10.1364/OL.28.000801)
    """

    def __init__(self, *args, zrange=None, **kwargs):
        """zrange : array-like
            An alternate way to specify the z range for the calculation
            must be expressed in the same units as wavelength
        """
        super().__init__(*args, **kwargs)
        if zrange is None:
            self._gen_zrange()
        else:
            self.zrange = zrange

    # include parent documentation
    __init__.__doc__ = BasePSF.__init__.__doc__ + __init__.__doc__

    def __repr__(self):
        return super().__repr__()[:-1] + f", zrange={self.zrange!r})"

    def _gen_zrange(self):
        """Internal utility to generate the zrange from zsize and zres"""
        self.zrange = (np.arange(self.zsize) - (self.zsize + 1) // 2) * self.zres

    @BasePSF.zsize.setter
    def zsize(self, value):
        # we need override this setter so that the zrange is recalculated
        BasePSF.zsize.fset(self, value)
        # try and except is necessary for initialization
        try:
            self._gen_zrange()
        except AttributeError:
            pass

    @BasePSF.zres.setter
    def zres(self, value):
        # same as for zsize
        BasePSF.zres.fset(self, value)
        try:
            self._gen_zrange()
        except AttributeError:
            pass

    @property
    def zrange(self):
        """The range overwhich to calculate the psf"""
        return self._zrange

    @zrange.setter
    def zrange(self, value):
        self._zrange = np.asarray(value)
        # check if passed value is scalar
        if not self._zrange.shape:
            # convert to array for later multiplications
            self._zrange.shape = (1,)
        self._attribute_changed()

    def _gen_kr(self):
        """Internal utiltiy to generate coordinate system and other internal
        parameters"""
        k = self._k = fftfreq(self.size, self.res)
        kxx, kyy = np.meshgrid(k, k)
        self._kr, self._phi = cart2pol(kyy, kxx)
        # kmag is the radius of the spherical shell of the OTF
        self._kmag = self.ni / self.wl
        # because the OTF only exists on a spherical shell we can calculate
        # a kz value for any pair of kx and ky values
        self._kz = psqrt(self._kmag ** 2 - self._kr ** 2)

    def _gen_pupil(self):
        """Generate an ideal pupil"""
        kr = self._kr
        # define the diffraction limit
        # remember we"re working with _coherent_ data _not_ intensity,
        # so drop the factor of 2
        diff_limit = self._na / self._wl
        # return a circle of intensity 1 over the ideal passband of the
        # objective make sure data is complex
        return (kr < diff_limit).astype(complex)

    def _calc_defocus(self):
        """Calculate the defocus to apply to the base pupil"""
        kz = self._kz
        return np.exp(2 * np.pi * 1j * kz * self.zrange[:, np.newaxis, np.newaxis])

    def _gen_psf(self, pupil_base=None):
        """An internal utility that generates the PSF
        Kwargs
        ------
        pupil_base : ndarray
            provided so that phase retrieval algorithms can hook into this
            method.

        NOTE: that the internal state is created with fftfreq, which creates
        _unshifted_ frequences"""
        # clear internal state
        self._attribute_changed()
        # generate internal coordinates
        self._gen_kr()
        # generate the pupil
        if pupil_base is None:
            pupil_base = self._gen_pupil()
        else:
            assert pupil_base.ndim == 2, f"`pupil_base` is wrong shape: {pupil_base.shape}"
            # Maybe we should do ifftshift here so user doesn't have too
        # pull relevant internal state variables
        kr = self._kr
        phi = self._phi
        kmag = self._kmag
        # apply the defocus to the base_pupil
        pupil = pupil_base * self._calc_defocus()
        # calculate theta, this is possible because we know that the
        # OTF is only non-zero on a spherical shell
        theta = np.arcsin((kr < kmag) * kr / kmag)
        # The authors claim that the following code is unecessary as the
        # sine condition is already taken into account in the definition
        # of the pupil, but I call bullshit
        if self.condition != "none":
            if self.condition == "sine":
                a = 1.0 / np.sqrt(np.cos(theta))
            elif self.condition == "herschel":
                a = 1.0 / np.cos(theta)
            else:
                raise RuntimeError("You should never see this")
            pupil *= a
        # apply the vectorial corrections, if requested
        if self.vec_corr != "none":
            plist = []
            if self.vec_corr == "z" or self.vec_corr == "total":
                plist.append(np.sin(theta) * np.cos(phi))  # Pzx
                plist.append(np.sin(theta) * np.sin(phi))  # Pzy
            if self.vec_corr == "y" or self.vec_corr == "total":
                plist.append((np.cos(theta) - 1) * np.sin(phi) * np.cos(phi))  # Pyx
                plist.append(np.cos(theta) * np.sin(phi) ** 2 + np.cos(phi) ** 2)  # Pyy
            if self.vec_corr == "x" or self.vec_corr == "total":
                plist.append(np.cos(theta) * np.cos(phi) ** 2 + np.sin(phi) ** 2)  # Pxx
                plist.append((np.cos(theta) - 1) * np.sin(phi) * np.cos(phi))  # Pxy
            # apply the corrections to the base pupil
            pupils = pupil * np.array(plist)[:, np.newaxis]
        else:
            # if no correction we still need one more axis for the following
            # code to work generally
            pupils = pupil[np.newaxis]
        # save the pupil for inspection, not necessary
        # self._pupils = pupils
        # because the internal state is created with fftfreq, no initial shift
        # is necessary.
        PSFa = fftshift(ifftn(pupils, axes=(2, 3)), axes=(2, 3))
        # save the PSF internally
        return PSFa

    def apply_pupil(self, pupil):
        """Apply a pupil function to the model"""
        self._attribute_changed()
        self.PSFa = self._gen_psf(pupil)

    @cached_property
    def OTFa(self):
        return easy_fft(self.PSFa, axes=(1, 2, 3))

    @cached_property
    def PSFa(self):
        return self._gen_psf()


class SheppardPSF(BasePSF):
    """Based on the following work

    [(1) Arnison, M. R.; Sheppard, C. J. R. A 3D Vectorial Optical Transfer
    Function Suitable for Arbitrary Pupil Functions. Optics Communications
    2002, 211 (1–6), 53–63.](dx.doi.org/10.1016/S0030-4018(02)01857-6)
    """

    dual = NumericProperty(attr="_dual", vartype=bool, doc="Simulate dual objectives")

    def __init__(self, *args, dual=False, **kwargs):
        """dual : bool
            Simulate dual objectives
        """
        super().__init__(*args, **kwargs)
        self.dual = dual

    # include parent documentation
    __init__.__doc__ = BasePSF.__init__.__doc__ + __init__.__doc__

    def __repr__(self):
        return super().__repr__()[:-1] + f", dual={self.dual})"

    @property
    def dual(self):
        """Simulate opposing objectives?"""
        return self._dual

    @dual.setter
    def dual(self, value):
        if not isinstance(value, bool):
            raise TypeError("`dual` must be a boolean")
        self._dual = value
        self._attribute_changed()

    @BasePSF.zres.setter
    def zres(self, value):
        # this checks the nyquist limit for z
        # remember that because we create a spherical shell for
        # The amplitude OTF not nyquist for the final intensity OTF ...
        max_val = self.wl / 2 / self.ni
        if value >= max_val:
            # this will cause a fftconvolution error when calculating the
            # intensity OTF
            raise ValueError(f"{value} is too large try a number smaller than {max_val}")
        BasePSF.zres.fset(self, value)

    def _gen_kr(self):
        """Internal utility function to generate internal state"""
        # generate internal kspace coordinates
        k = fftfreq(self.size, self.res)
        kz = fftfreq(self.zsize, self.zres)
        k_tot = np.meshgrid(kz, k, k, indexing="ij")
        # calculate r
        kr = norm(k_tot, axis=0)
        # calculate the radius of the spherical shell in k-space
        self.kmag = kmag = self.ni / self.wl
        # determine k-space pixel size
        dk, dkz = k[1] - k[0], kz[1] - kz[0]
        # save output for user
        self.dk, self.dkz = dk, dkz
        # determine the min value for kz given the NA and wavelength
        kz_min = np.sqrt(kmag ** 2 - (self.na / self.wl) ** 2)
        # make sure we're not crazy
        assert kz_min >= 0, "Something went horribly wrong"
        # if the user gave us different z and x/y res we need to calculate
        # the positional "error" in k-space to draw the spherical shell
        if dk != dkz:
            with np.errstate(invalid="ignore"):
                dd = np.array((dkz, dk, dk)).reshape(3, 1, 1, 1)
                dkr = norm(np.array(k_tot) * dd, axis=0) / kr
            # we know the origin is zero so replace it
            dkr[0, 0, 0] = 0.0
        else:
            dkr = dk
        if self.dual:
            # if we want dual objectives we need two spherical shells
            kzz = abs(k_tot[0])
        else:
            kzz = k_tot[0]
        # calculate the points on the spherical shell, save them and the
        # corresponding kz, ky and kx coordinates
        self.valid_points = np.logical_and(abs(kr - kmag) < dkr, kzz > kz_min + dkr)
        self.kzz, self.kyy, self.kxx = [k[self.valid_points] for k in k_tot]

    def _gen_otf(self):
        """Internal utility function to generate the OTFs"""
        # clear internal state
        self._attribute_changed()
        # generate coordinate space
        self._gen_kr()
        kxx, kyy, kzz = self.kxx, self.kyy, self.kzz
        # generate direction cosines
        m, n, s = np.array((kxx, kyy, kzz)) / norm((kxx, kyy, kzz), axis=0)
        # apply a given imaging condition
        if self.condition == "sine":
            a = 1.0 / np.sqrt(s)
        elif self.condition == "herschel":
            a = 1.0 / s
        elif self.condition == "none":
            a = 1.0
        else:
            raise RuntimeError("You should never see this")
        # apply the vectorial corrections if requested
        if self.vec_corr != "none":
            plist = []
            if self.vec_corr == "z" or self.vec_corr == "total":
                plist.append(-m)  # Pzx
                plist.append(-n)  # Pzy
            if self.vec_corr == "y" or self.vec_corr == "total":
                plist.append(-n * m / (1 + s))  # Pyx
                plist.append(1 - n ** 2 / (1 + s))  # Pyy
            if self.vec_corr == "x" or self.vec_corr == "total":
                plist.append(1 - m ** 2 / (1 + s))  # Pxx
                plist.append(-m * n / (1 + s))  # Pxy
            # generate empty otf
            otf = np.zeros((len(plist), self.zsize, self.size, self.size), dtype="D")
            # fill in the valid poins
            for o, p in zip(otf, plist):
                o[self.valid_points] = p * a
        else:
            # TODO: we can actually do a LOT better here.
            # if the vectorial correction is None then we can
            # calculate a 2D (kz, kr) OTF and interpolate it out to
            # the full 3D size.
            # otf_sub = self._gen_radsym_otf()
            # otf = otf_sub[np.newaxis]
            otf_sub = np.zeros((self.zsize, self.size, self.size), dtype="D")
            otf_sub[self.valid_points] = 1.0
            otf = otf_sub[np.newaxis]
        # we're already calculating the OTF, so we just need to shift it into
        # the right place.
        return fftshift(otf, axes=(1, 2, 3))

    @cached_property
    def OTFa(self):
        return self._gen_otf()

    @cached_property
    def PSFa(self):
        return easy_ifft(self.OTFa, axes=(1, 2, 3))


def apply_aberration(model, mcoefs, pcoefs):
    """Applies a set of abberations to a model PSF

    Parameters
    ----------
    model : HanserPSF
        The model PSF to which to apply the aberrations
    mcoefs : ndarray (n, )
        The magnitude coefficiencts
    pcoefs : ndarray (n, )
        The phase coefficients
    
    Note: this function assumes the mcoefs and pcoefs are Noll ordered"""

    # sanity checks
    assert isinstance(model, HanserPSF), "Model must be a HanserPSF"

    model = copy.copy(model)

    if mcoefs is None and pcoefs is None:
        logger.warning("No abberation applied")
        return model

    if mcoefs is None:
        mcoefs = np.zeros_like(pcoefs)

    if pcoefs is None:
        pcoefs = np.zeros_like(mcoefs)

    assert len(mcoefs) == len(pcoefs), "Coefficient lengths don't match"

    # extract kr
    model._gen_kr()
    kr = model._kr
    theta = model._phi
    # make zernikes (need to convert kr to r where r = 1 when kr is at
    # diffraction limit)
    r = kr * model.wl / model.na
    zerns = zernike(r, theta, np.arange(len(mcoefs)) + 1)

    pupil_phase = (zerns * pcoefs[:, None, None]).sum(0)
    pupil_mag = (zerns * mcoefs[:, None, None]).sum(0)

    # apply aberrations to unaberrated pupil (N.B. the unaberrated phase is 0)
    pupil_mag += abs(model._gen_pupil())

    # generate the PSF, assign to attribute
    pupil_total = pupil_mag * np.exp(1j * pupil_phase)
    model.apply_pupil(pupil_total)

    return model


def apply_named_aberration(model, aberration, magnitude):
    """A convenience function to apply a specific named aberration to the PSF. This will only effect the phase"""
    # get the Noll number and build pcoefs
    try:
        noll = name2noll[aberration]
    except KeyError as e:
        raise KeyError(
            f"Aberration '{aberration}' unknown, choose from: '"
            + "', '".join(name2noll.keys())
            + "'"
        )
    pcoefs = np.zeros(noll)
    pcoefs[-1] = magnitude
    return apply_aberration(model, None, pcoefs)


if __name__ == "__main__":
    # import plotting
    from matplotlib import pyplot as plt

    # generate a comparison
    kwargs = dict(
        wl=520,
        na=1.27,
        ni=1.33,
        res=90,
        size=256,
        zres=190,
        zsize=128,
        vec_corr="none",
        condition="none",
    )
    psfs = HanserPSF(**kwargs), SheppardPSF(**kwargs)

    with plt.style.context("dark_background"):

        fig, axs = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw=dict(width_ratios=(1, 2)))

        for psf, ax_sub in zip(psfs, axs):
            print(psf)
            # make coordinates
            ax_yx, ax_zx = ax_sub
            # get magnitude
            otf = abs(psf.OTFi)
            # normalize
            otf /= otf.max()
            otf /= otf.mean()
            otf = np.log(otf + np.finfo(float).eps)

            # plot
            style = dict(vmin=-3, vmax=5, cmap="inferno", interpolation="bicubic")
            ax_yx.matshow(otf[otf.shape[0] // 2], **style)
            ax_yx.set_title("{} $k_y k_x$ plane".format(psf.__class__.__name__))
            ax_zx.matshow(otf[..., otf.shape[1] // 2], **style)
            ax_zx.set_title("{} $k_z k_x$ plane".format(psf.__class__.__name__))

            for ax in ax_sub:
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
        fig.tight_layout()

    # NOTE: the results are _very_ close on a qualitative scale, but they do not match exactly
    # as theory says they should (they're mathematically identical to one another)

    model_kwargs = dict(
        wl=525, na=1.27, ni=1.33, res=70, size=256, zrange=[0], vec_corr="none", condition="none",
    )
    model = HanserPSF(**model_kwargs)

    mag = model.na / model.wl * model.res * 2 * np.pi

    with plt.style.context("dark_background"):
        fig, axs = plt.subplots(3, 5, figsize=(12, 8))
        # fill out plot
        for ax, name in zip(axs.ravel(), name2noll.keys()):
            model2 = apply_named_aberration(model, name, mag * 2)
            ax.imshow(
                model2.PSFi.squeeze()[104:-104, 104:-104], cmap="inferno", interpolation="bicubic"
            )
            ax.set_xlabel(name.replace(" ", "\n", 1).title())
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        # fig.tight_layout()
    plt.show()
