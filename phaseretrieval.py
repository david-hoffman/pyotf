#!/usr/bin/env python
# -*- coding: utf-8 -*-
# phaseretrieval.py
"""
Back focal plane (pupil) phase retrieval algorithm base on:
[(1) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
Phase Retrieval for High-Numerical-Aperture Optical Systems.
Optics Letters 2003, 28 (10), 801.](dx.doi.org/10.1364/OL.28.000801)

Copyright (c) 2016, David Hoffman
"""
import copy
import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import ifftshift, fftshift, fftn
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import ifftshift, fftshift, fftn

from numpy.linalg import lstsq
from .utils import *
from . import HanserPSF
from .zernike import *
from skimage.restoration import unwrap_phase
from matplotlib import pyplot as plt


def retrieve_phase(data, params, max_iters=200, pupil_tol=1e-8,
                   mse_tol=1e-8, phase_only=False):
    """Code that actually runs the phase retrieval

    Parameters
    ----------
    data : ndarray (3 dim)
        The experimentally measured data
    params : dict
        Parameters to pass to HanserPSF, size and zsize will be automatically
        updated from data.shape
    max_iters : int
        The maximum number of iterations to run, default is 200
    pupil_tol : float
        the tolerance in percent change in change in pupil, default is 1e-8
    mse_tol : float
        the tolerance in percent change for the mean squared error between
        data and simulated data
    """
    # make sure data is square
    assert data.shape[1] == data.shape[2], "Data is not square in x/y"
    assert data.ndim == 3, "Data doesn't have enough dims"
    # make sure the user hasn't screwed up the params
    params.update(dict(vec_corr="none", condition="none"))
    # assume that data prep has been handled outside function
    # The field magnitude is the square root of the intensity
    mag = psqrt(data)
    # generate a model from parameters
    model = HanserPSF(**params)
    model._gen_kr()
    # start a list for iteration
    mse = np.zeros(max_iters)
    mse_diff = np.zeros(max_iters)
    pupil_diff = np.zeros(max_iters)
    # generate a pupil to start with
    new_pupil = model._gen_pupil()
    # save it as a mask
    mask = new_pupil.real
    # iterate
    for i in range(max_iters):
        # generate new mse and add it to the list
        model._gen_psf(new_pupil)
        new_mse = _calc_mse(data, model.PSFi)
        mse[i] = new_mse
        if i > 0:
            # calculate the difference in mse to test for convergence
            mse_diff[i] = abs(old_mse - new_mse) / old_mse
            # calculate the difference in pupil
            pupil_diff[i] = (abs(old_pupil - new_pupil)**2).mean() / (abs(old_pupil)**2).mean()
        else:
            mse_diff[i] = np.nan
            pupil_diff[i] = np.nan
        # check tolerances
        if pupil_diff[i] < pupil_tol or mse_diff[i] < mse_tol:
            break
        # update old_mse
        old_mse = new_mse
        # retrieve new pupil
        old_pupil = new_pupil
        # keep phase
        phase = np.angle(model.PSFa.squeeze())
        # replace magnitude with experimentally measured mag
        new_psf = mag * np.exp(1j * phase)
        # generate the new pupils
        new_pupils = fftn(fftshift(new_psf, axes=(1, 2)), axes=(1, 2))
        # undo defocus and take the mean
        new_pupils /= model._calc_defocus()
        new_pupil = new_pupils.mean(0) * mask
        # if phase only discard magnitude info
        if phase_only:
            new_pupil = np.exp(1j * np.angle(new_pupil)) * mask
    mse = mse[:i + 1]
    mse_diff = mse_diff[:i + 1]
    pupil_diff = pupil_diff[:i + 1]
    # shift mask
    mask = ifftshift(mask)
    # shift phase then unwrap and mask
    phase = unwrap_phase(ifftshift(np.angle(old_pupil))) * mask
    # shift magnitude
    magnitude = ifftshift(abs(old_pupil)) * mask
    return PhaseRetrievalResult(magnitude, phase, np.array(mse), pupil_diff,
                                mse_diff, model)


class PhaseRetrievalResult(object):
    """An object for holding the result of phase retrieval"""

    def __init__(self, mag, phase, mse, pupil_diff, mse_diff, model):
        self.mag = mag
        self.phase = phase
        self.mse = mse
        self.pupil_diff = pupil_diff
        self.mse_diff = mse_diff
        self.model = model
        model._gen_kr()
        r, theta = model._kr, model._phi
        self.r, self.theta = ifftshift(r), ifftshift(theta)
        self.na, self.wl = model.na, model.wl

    def fit_to_zernikes(self, num_zerns):
        """Fits the data to a number of zernikes"""
        # set up coordinate system
        r, theta = self.r, self.theta
        r = r / (self.na / self.wl)
        zerns = zernike(r, theta, np.arange(1, num_zerns + 1))
        mag_coefs = _fit_to_zerns(self.mag, zerns)
        phase_coefs = _fit_to_zerns(self.phase, zerns)
        self.zd_result = ZernikeDecomposition(mag_coefs, phase_coefs, zerns)

    def generate_psf(self, sphase=slice(4, None, None), size=None, zsize=None,
                     zrange=None):
        """Make a perfect PSF"""
        # instead of going through all this trouble, why not take original
        # model and pad out with zeros? Effect should be nearly the same with
        # out this complexity and time cost.
        pcoefs = self.zd_result.pcoefs
        model = copy.copy(self.model)
        if size is not None:
            model.size = size
        if zsize is not None:
            model.zsize = zsize
        if zrange is not None:
            model.zrange = zrange
        model._gen_kr()
        r, theta = model._kr, model._phi
        r = r / (self.na / self.wl)
        new_zerns = zernike(r, theta, np.arange(1, pcoefs.size + 1))
        new_zd_result = ZernikeDecomposition(self.zd_result.mcoefs,
                                             self.zd_result.pcoefs, new_zerns)
        model._gen_psf(new_zd_result.complex_pupil(sphase=sphase))
        return model.PSFi

    def plot(self):
        """Plot the retrieved results"""
        fig, (ax_phase, ax_mag) = plt.subplots(1, 2, figsize=(12, 5))
        phase_img = ax_phase.matshow(self.phase, cmap="seismic")
        plt.colorbar(phase_img, ax=ax_phase)
        mag_img = ax_mag.matshow(self.mag, cmap="inferno")
        plt.colorbar(mag_img, ax=ax_mag)
        ax_phase.set_title("Pupil Phase")
        ax_mag.set_title("Pupil Magnitude")
        fig.tight_layout()
        return fig, (ax_phase, ax_mag)

    def plot_convergence(self):
        """Diagnostic plots of the convergence criteria"""
        fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        for ax, data in zip(axs, (self.mse, self.mse_diff, self.pupil_diff)):
            with np.errstate(invalid="ignore"):
                ax.semilogy(data)
        for ax, t in zip(axs, ("Mean Squared Error",
                               "Relative Change in MSE",
                               "Relative Change in Pupil")):
            ax.set_title(t)
        fig.tight_layout()
        return fig, axs

    @property
    def complex_pupil(self):
        """Return the complex pupil function"""
        return self.mag * np.exp(1j * self.phase)


class ZernikeDecomposition(object):
    """An object for holding the results of a zernike decomposition"""

    def __init__(self, mag_coefs, phase_coefs, zerns):
        self.mcoefs = mag_coefs
        self.pcoefs = phase_coefs
        self.zerns = zerns

    def plot_coefs(self):
        fig, axs = plt.subplots(2, 1, sharex=True)
        for ax, data in zip(axs, (self.mcoefs, self.pcoefs)):
            ax.bar(np.arange(data.size) + 1, data)
            ax.axis("tight")
        for ax, t in zip(axs, ("Magnitude Coefficients",
                               "Phase Coefficients")):
            ax.set_title(t)
        ax.set_xlabel("Noll's Number")
        fig.tight_layout()
        return fig, axs

    def plot_named_coefs(self):
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 6))
        ordered_names = [noll2name[i + 1] for i in range(len(noll2name))]
        x = np.arange(len(ordered_names)) + 1
        data = self.pcoefs[:len(ordered_names)]
        ax.bar(x, data, align="center", tick_label=ordered_names)
        ax.axis("tight")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylabel("Phase Coefficient")
        ax.set_xlabel("Noll's Number")
        fig.tight_layout()
        return fig, ax

    def _recon(self, coefs, s=Ellipsis):
        """reconstruct mag or phase"""
        return _recon_from_zerns(coefs[s], self.zerns[s])

    def phase(self, *args, **kwargs):
        """Reconstruct the phase from the specified slice"""
        return self._recon(self.pcoefs, *args, **kwargs)

    def mag(self, *args, **kwargs):
        """Reconstruct the magnitude from the specified slice"""
        return self._recon(self.mcoefs, *args, **kwargs)

    def complex_pupil(self, smag=Ellipsis, sphase=Ellipsis, *args, **kwargs):
        """Reconstruct the complex pupil from the specified slice"""
        return self.mag(*args, s=smag, **kwargs) * np.exp(1j * self.phase(*args, s=sphase, **kwargs))


def _calc_mse(data1, data2):
    """utility to calculate mean square error"""
    return ((data1 - data2) ** 2).mean()


def _fit_to_zerns(data, zerns, **kwargs):
    """sub function that does the reshaping and the least squares"""
    data2fit = data.reshape(-1)
    zerns2fit = zerns.reshape(zerns.shape[0], -1).T
    coefs, _, _, _ = lstsq(zerns2fit, data2fit, **kwargs)
    return coefs


def _recon_from_zerns(coefs, zerns):
    """Utility to reconstruct from coefs"""
    return (coefs[:, np.newaxis, np.newaxis] * zerns).sum(0)
