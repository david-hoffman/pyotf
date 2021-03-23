#!/usr/bin/env python
# -*- coding: utf-8 -*-
# figures.py
"""
Simple script to generate the figures in the README.md.

Copyright (c) 2020, David Hoffman
"""

import time
import warnings

import numpy as np
from matplotlib import pyplot as plt

import tifffile as tif

from pyotf.otf import HanserPSF, SheppardPSF, apply_named_aberration
from pyotf.zernike import zernike, cart2pol, noll2name, noll2degrees, name2noll
from pyotf.phaseretrieval import retrieve_phase
from pyotf.utils import prep_data_for_PR

OTF_MODEL = dict(
    wl=525,  # units in nm
    na=1.27,
    ni=1.33,
    res=90,
    size=256,
    zres=190,
    zsize=128,
    vec_corr="none",
    condition="none",
)

SAVE = dict(dpi=150, transparent=False, bbox_inches="tight")


def otf_plots(model_kwargs):
    """Make OTF plots.
    
    NOTE: the results are _very_ close on a qualitative scale, but they do not match exactly as
    theory says they should (they're mathematically identical to one another)
    """
    # generate a comparison

    psfs = HanserPSF(**model_kwargs), SheppardPSF(**model_kwargs)

    fig, axs = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw=dict(width_ratios=(1, 2)))

    for psf, ax_sub in zip(psfs, axs):
        print(f"Making {psf} plot")
        # make coordinates
        ax_yx, ax_zx = ax_sub
        # get magnitude
        otf = abs(psf.OTFi)
        # normalize
        otf /= otf.max()
        otf /= otf.mean()
        otf = np.log(otf + np.finfo(float).eps)

        # plot
        ax_yx.imshow(
            otf[otf.shape[0] // 2], vmin=-3, vmax=5, cmap="inferno", interpolation="bicubic"
        )
        ax_yx.set_title("{} $k_y k_x$ plane".format(psf.__class__.__name__))
        ax_zx.imshow(
            otf[..., otf.shape[1] // 2], vmin=-3, vmax=5, cmap="inferno", interpolation="bicubic"
        )
        ax_zx.set_title("{} $k_z k_x$ plane".format(psf.__class__.__name__))

        for ax in ax_sub:
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    fig.tight_layout()
    fig.savefig("fixtures/otf.png", **SAVE)


def aberration_plots(model_kwargs):
    """Make aberration plots."""
    model_kwargs = model_kwargs.copy()
    model_kwargs["zrange"] = [0]
    model_kwargs["vec_corr"] = "total"
    model_kwargs["condition"] = "sine"
    model = HanserPSF(**model_kwargs)

    mag = model.na / model.wl * model.res * 2 * np.pi

    fig, axs = plt.subplots(3, 5, figsize=(12, 8))
    # fill out plot
    for ax, name in zip(axs.ravel(), name2noll.keys()):
        print(f"Making {name} plot")
        model2 = apply_named_aberration(model, name, mag * 2)
        ax.imshow(
            model2.PSFi.squeeze()[104:-104, 104:-104], cmap="inferno", interpolation="bicubic"
        )
        ax.set_xlabel(name.replace(" ", "\n", 1).title())
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    fig.savefig("fixtures/aberrations.png", **SAVE)


def zernike_plots():
    """Make zernike plots."""
    # make coordinates
    x = np.linspace(-1, 1, 1025)
    xx, yy = np.meshgrid(x, x)  # xy indexing is default
    r, theta = cart2pol(yy, xx)
    # set up plot
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    # fill out plot
    for ax, (k, v) in zip(axs.ravel(), noll2name.items()):
        print(f"Making {v} plot")
        zern = zernike(r, theta, k)
        ax.imshow(
            np.ma.array(zern, mask=r > 1),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            interpolation="bicubic",
        )
        ax.set_title(v + r", $Z_{{{}}}^{{{}}}$".format(*noll2degrees(k)))
        ax.axis("off")
    fig.tight_layout()

    fig.savefig("fixtures/zernike.png", **SAVE)


def pr_plots():
    """Make phase retrieval plots."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = tif.imread("fixtures/psf_wl520nm_z300nm_x130nm_na0.85_n1.0.tif")

    # prep data
    data_prepped = prep_data_for_PR(data, 512, 1.1)

    # set up model params
    params = dict(wl=520, na=0.85, ni=1.0, res=130, zres=300)

    pr_result = retrieve_phase(data_prepped, params, 200, 1e-6, 1e-6)

    # plot
    fig, axs = pr_result.plot()
    fig.savefig("fixtures/PR Result.png", **SAVE)
    fig, axs = pr_result.plot_convergence()
    fig.savefig("fixtures/PR Convergence.png", **SAVE)

    # fit to zernikes
    pr_result.fit_to_zernikes(120)

    # plot
    fig, axs = pr_result.zd_result.plot_named_coefs()
    fig.savefig("fixtures/Named Coefs.png", **SAVE)
    pr_result.zd_result.plot_coefs()

    fig, axs = pr_result.zd_result.plot()
    fig.savefig("fixtures/PR Result ZD.png", **SAVE)


if __name__ == "__main__":
    otf_plots(OTF_MODEL)
    zernike_plots()
    aberration_plots(OTF_MODEL)
    pr_plots()
