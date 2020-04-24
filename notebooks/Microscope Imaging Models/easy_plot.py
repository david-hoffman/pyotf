#!/usr/bin/env python
# -*- coding: utf-8 -*-
# easy_plot.py
"""
An easy plotting function

Copyright (c) 2020, David Hoffman
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np

from pyotf.utils import easy_fft, easy_ifft
from dphutils import bin_ndarray

# plot function 😬
def easy_plot(psfs, labels, oversample_factor=1, res=1, gam=0.3, vmin=1e-3):
    ncols = len(psfs)

    assert ncols == len(labels), "Lengths mismatched"
    assert ncols < 10

    plot_size = 2.0

    fig = plt.figure(None, (plot_size * ncols, plot_size * 4), dpi=150)
    grid = ImageGrid(fig, 111, nrows_ncols=(4, ncols), axes_pad=0.1)

    fig2, axp = plt.subplots(dpi=150, figsize=(plot_size * ncols, 4))

    for (i, p), l, col in zip(enumerate(psfs), labels, grid.axes_column):
        p = bin_ndarray(p, bin_size=oversample_factor)
        p /= p.max()
        col[0].imshow(p.max(1), norm=mpl.colors.PowerNorm(gam), interpolation="bicubic")
        col[1].imshow(p.max(0), norm=mpl.colors.PowerNorm(gam), interpolation="bicubic")

        col[0].set_title(l)

        otf = abs(easy_fft(p))
        otf /= otf.max()
        otf = np.fmax(otf, vmin)
        c = (len(otf) + 1) // 2

        col[2].matshow(otf[:, c], norm=mpl.colors.LogNorm(), interpolation="bicubic")
        col[3].matshow(otf[c], norm=mpl.colors.LogNorm(), interpolation="bicubic")

        pp = p[:, c, c]
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
