#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py
"""
A module to simulate optical transfer functions and point spread functions

If this file is run as a script (python -m pyOTF.otf) it will compare
the HanserPSF to the SheppardPSF in a plot.

https://en.wikipedia.org/wiki/Optical_transfer_function
https://en.wikipedia.org/wiki/Point_spread_function

Copyright (c) 2016, David Hoffman
"""

import numpy as np
from numpy.linalg import norm
try:
    from pyfftw.interfaces.numpy_fft import fftshift, fftfreq
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fftshift, fftfreq
from pyOTF.utils import *


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
    wl = NumericProperty(attr="_wl", vartype=(float, int),
                         doc="Wavelength of emission, in nm")
    na = NumericProperty(attr="_na", vartype=(float, int),
                         doc="Numerical Aperature")
    ni = NumericProperty(attr="_ni", vartype=(float, int),
                         doc="Refractive index")
    size = NumericProperty(attr="_size", vartype=int, doc="x/y size")
    zsize = NumericProperty(attr="_zsize", vartype=int, doc="z size")

    def __init__(self, wl, na, ni, res, size, zres=None, zsize=None,
                 vec_corr="none", condition="sine"):
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
            keyword to indicate whether to model the sine or herschel
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

    def _attribute_changed(self):
        """Called whenever key attributes are changed
        Sets internal state variables to None so that when the
        user asks for them they are recalculated"""
        self._PSFi = None
        self._PSFa = None
        self._OTFi = None
        self._OTFa = None

    @property
    def zres(self):
        """z resolution (nm)"""
        return self._zres

    @zres.setter
    def zres(self, value):
        # make sure z is positive
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
        # max_val is the nyquist limit, for an accurate simulation
        # the pixel size must be smaller than this number
        # thinking in terms of the convolution that is implicitly
        # performed when generating the OTFi we also don't want
        # any wrapping effects.
        max_val = 1 / (2 * self.na / self.wl) / 2
        if value >= max_val:
            raise ValueError(
                ("{!r} is larger than the Nyquist Limit,"
                 " try a number smaller than {!r}").format(
                    value, max_val)
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
            raise ValueError(
                ("Vector correction must be one of "
                 ("{!r}, " * len(valid_values)).format(value))
            )
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
            raise ValueError(
                ("Condition must be one of "
                 ("{!r}, " * len(valid_values)).format(value))
            )
        self._condition = value
        self._attribute_changed()

    @property
    def OTFa(self):
        """Amplitude OTF (coherent transfer function), complex array"""
        raise NotImplementedError

    @property
    def PSFa(self):
        """Amplitude PSF, complex array"""
        raise NotImplementedError

    @property
    def PSFi(self):
        """Intensity PSF, real array"""
        if self._PSFi is None:
            # the intensity PSFs are the absolute value of the coherent PSF
            # because our imaging is _incoherent_ the result is simply the sum
            # of the intensities for each vectorial component.
            self._PSFi = (abs(self.PSFa)**2).sum(axis=0)
        return self._PSFi

    @property
    def OTFi(self):
        """Intensity OTF, complex array"""
        if self._OTFi is None:
            self._OTFi = easy_fft(self.PSFi)
        return self._OTFi


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

    def _gen_zrange(self):
        """Internal utility to generate the zrange from zsize and zres"""
        self.zrange = (np.arange(self.zsize) - (self.zsize - 1) / 2) * self.zres

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
            self._zrange.shape = (1, )
        self._attribute_changed()

    def _gen_kr(self):
        """Internal utiltiy to generate coordinate system and other internal
        parameters"""
        k = fftfreq(self.size, self.res)
        kxx, kyy = np.meshgrid(k, k)
        self._kr, self._phi = cart2pol(kyy, kxx)
        # kmag is the radius of the spherical shell of the OTF
        self._kmag = self.ni / self.wl
        # because the OTF only exists on a spherical shell we can calculate
        # a kz value for any pair of kx and ky values
        self._kz = psqrt(self._kmag**2 - self._kr**2)

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
        return np.exp(2 * np.pi * 1j * kz *
                      self.zrange[:, np.newaxis, np.newaxis])

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
            assert pupil_base.ndim == 2, "`pupil_base` is wrong shape"
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
        if self.condition == "sine":
            a = 1.0 / np.sqrt(np.cos(theta))
        elif self.condition == "herschel":
            a = 1.0 / np.cos(theta)
        elif self.condition == "none":
            a = 1.0
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
                plist.append(np.cos(theta) * np.sin(phi)**2 + np.cos(phi)**2)  # Pyy
            if self.vec_corr == "x" or self.vec_corr == "total":
                plist.append(np.cos(theta) * np.cos(phi)**2 + np.sin(phi)**2)  # Pxx
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
        self._PSFa = PSFa

    # Because the _attribute_changed() method sets all the internal OTFs and
    # PSFs None we can recalculate them only when needed
    @property
    def OTFa(self):
        if self._OTFa is None:
            self._OTFa = easy_fft(self.PSFa, axes=(1, 2, 3))
        return self._OTFa

    @property
    def PSFa(self):
        if self._PSFa is None:
            self._gen_psf()
        return self._PSFa


class SheppardPSF(BasePSF):
    """Based on the following work

    [(1) Arnison, M. R.; Sheppard, C. J. R. A 3D Vectorial Optical Transfer
    Function Suitable for Arbitrary Pupil Functions. Optics Communications
    2002, 211 (1–6), 53–63.](dx.doi.org/10.1016/S0030-4018(02)01857-6)
    """

    def __init__(self, *args, dual=False, **kwargs):
        """dual : bool
            Simulate dual objectives
        condition : str
            Keyword indicating imaging condition
                Valid values: "none", "sine", "herschel"
        """
        super().__init__(*args, **kwargs)
        self.dual = dual

    # include parent documentation
    __init__.__doc__ = BasePSF.__init__.__doc__ + __init__.__doc__

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
        max_val = 1 / (2 * self.ni / self.wl)
        if value >= max_val:
            # this will cause a fftconvolution error when calculating the
            # intensity OTF
            raise ValueError(
                "{!r} is too large try a number smaller than {!r}".format(
                    value, max_val)
            )
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
            with np.errstate(invalid='ignore'):
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
        self.valid_points = np.logical_and(abs(kr - kmag) < dkr,
                                           kzz > kz_min + dkr)
        self.kzz, self.kyy, self.kxx = [k[self.valid_points] for k in k_tot]

    def _gen_radsym_otf(self):
        """Generate a radially symmetric OTF first and then interpolate to
        requested size"""
        raise NotImplementedError

    def _gen_otf(self):
        """Internal utility function to generate the OTFs"""
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
            otf = np.zeros((len(plist), self.zsize, self.size, self.size),
                           dtype="D")
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
        self._OTFa = fftshift(otf, axes=(1, 2, 3))

    @property
    def OTFa(self):
        if self._OTFa is None:
            self._gen_otf()
        return self._OTFa

    @property
    def PSFa(self):
        if self._PSFa is None:
            self._PSFa = easy_ifft(self.OTFa, axes=(1, 2, 3))
        return self._PSFa


class SheppardPSF2D(SheppardPSF):
    """A two dimensional version of `SheppardPSF`"""

    def _gen_kr(self):
        """Internal utility function to generate internal state"""
        # generate internal kspace coordinates
        k = fftfreq(self.size, self.res)
        kz = fftfreq(self.zsize, self.zres)
        k_tot = np.meshgrid(kz, k, indexing="ij")
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
            with np.errstate(invalid='ignore'):
                dd = np.array((dkz, dk)).reshape(2, 1, 1)
                dkr = norm(np.array(k_tot) * dd, axis=0) / kr
            # we know the origin is zero so replace it
            dkr[0, 0] = 0.0
        else:
            dkr = dk
        if self.dual:
            # if we want dual objectives we need two spherical shells
            kzz = abs(k_tot[0])
        else:
            kzz = k_tot[0]
        # calculate the points on the spherical shell, save them and the
        # corresponding kz, ky and kx coordinates
        self.valid_points = np.logical_and(abs(kr - kmag) < dkr,
                                           kzz > kz_min + dkr)
        self.kzz, self.krr = [k[self.valid_points] for k in k_tot]

    def _gen_otf(self):
        """Internal utility function to generate the OTFs"""
        # generate coordinate space
        self._gen_kr()
        krr, kzz = self.krr, self.kzz
        # generate direction cosines
        n, s = np.array((krr, kzz)) / norm((krr, kzz), axis=0)
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
            otf = np.zeros((len(plist), self.zsize, self.size, self.size),
                           dtype="D")
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
            otf_sub = np.zeros((self.zsize, self.size), dtype="D")
            otf_sub[self.valid_points] = 1.0
            otf = otf_sub[np.newaxis]
        # we're already calculating the OTF, so we just need to shift it into
        # the right place.
        self._OTFa = fftshift(otf, axes=(1, 2))

    @property
    def PSFa(self):
        if self._PSFa is None:
            self._PSFa = easy_ifft(self.OTFa, axes=(1, 2))
        return self._PSFa


if __name__ == "__main__":
    # import plotting
    from matplotlib import pyplot as plt
    # generate a comparison
    args = (488, 0.85, 1.0, 140, 256, 240, 128)
    kwargs = dict(vec_corr="none", condition="none")
    psf = [HanserPSF(*args, **kwargs), SheppardPSF(*args, **kwargs)]
    with plt.style.context("dark_background"):
        fig, axs = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw=dict(width_ratios=(1, 2)))
        for p, ax_sub in zip(psf, axs):
            # make coordinates
            ax_yx, ax_zx = ax_sub
            otf = abs(p.OTFi)
            otf /= otf.max()
            otf /= otf.mean()
            otf = np.log(otf + np.finfo(float).eps)
            ax_yx.matshow(otf[otf.shape[0] // 2], vmin=-5, vmax=5, cmap="inferno")
            ax_yx.set_title("{} $k_y k_x$ plane".format(p.__class__.__name__))
            ax_zx.matshow(otf[..., otf.shape[1] // 2], vmin=-5, vmax=5,
                          cmap="inferno")
            ax_zx.set_title("{} $k_z k_x$ plane".format(p.__class__.__name__))
            for ax in ax_sub:
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())
        fig.tight_layout()
        plt.show()
    # NOTE: the results are _very_ close on a qualitative scale, but they
    # do not match exactly as theory says they should (they're
    # mathematically identical to one another)
