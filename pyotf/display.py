#!/usr/bin/env python
# -*- coding: utf-8 -*-
# display.py
"""
Display functions for PSFs and OTFs

Copyright (c) 2021, David Hoffman
"""

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def max_min(n, d):
    return np.array((-n // 2, (n - 1) // 2 + n % 2)) * d


def fft_max_min(n, d):
    step_size = 1 / d / n
    return max_min(n, step_size)


def add_scalebar(ax, scalebar_size, pixel_size, unit="µm", edgecolor=None, **kwargs):
    """Add a scalebar to the axis."""
    scalebar_length = scalebar_size / pixel_size
    default_scale_bar_kwargs = dict(
        loc="lower right",
        pad=0.5,
        color="white",
        frameon=False,
        size_vertical=scalebar_length / 10,
        fontproperties=fm.FontProperties(weight="bold"),
    )
    default_scale_bar_kwargs.update(kwargs)
    if unit is not None:
        label = f"{scalebar_size} {unit}"
    else:
        label = ""
        if "lower" in default_scale_bar_kwargs["loc"]:
            default_scale_bar_kwargs["label_top"] = True
    scalebar = AnchoredSizeBar(ax.transData, scalebar_length, label, **default_scale_bar_kwargs)
    if edgecolor:
        scalebar.size_bar.get_children()[0].set_edgecolor(edgecolor)
        scalebar.txt_label.get_children()[0].set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground=edgecolor), path_effects.Normal()]
        )
    # add the scalebar
    ax.add_artist(scalebar)
    return scalebar


def z_squeeze(n1, n2, na=0.85):
    """Amount z expands or contracts when using an objective designed
    for one index (n1) to image into a medium with another index (n2)"""

    if n1 == n2:
        return 1

    def func(n):
        return n - np.sqrt(max(0, n ** 2 - na ** 2))

    return func(n1) / func(n2)


def psf_plot(
    psf,
    *,
    na=0.85,
    nobj=1.0,
    nsample=None,
    zstep=0.25,
    pixel_size=0.13,
    fig=None,
    loc=111,
    mip=True,
    **kwargs,
):
    """"""
    if nsample is None:
        nsample = nobj
    # expand z step
    zstep *= z_squeeze(nobj, nsample, na)
    # update our default kwargs for plotting
    dkwargs = dict(interpolation="nearest", cmap="inferno")
    dkwargs.update(kwargs)
    # make the fig if one isn't passed
    if fig is None:
        fig = plt.figure(None, (8.0, 8.0))

    grid = mpl.ImageGrid(fig, loc, nrows_ncols=(2, 2), axes_pad=0.3)
    # calc extents
    nz, ny, nx = psf.shape
    kz, ky, kx = [max_min(n, d) for n, d in zip(psf.shape, (zstep, pixel_size, pixel_size))]

    # do plotting
    if mip:
        grid[3].imshow(psf.max(0), **dkwargs, extent=(*kx, *ky))
        grid[2].imshow(psf.max(1).T, **dkwargs, extent=(*kz, *ky))
        grid[1].imshow(psf.max(2), **dkwargs, extent=(*kx, *kz))
    else:
        grid[3].imshow(psf[nz // 2, :, :], **dkwargs, extent=(*kx, *ky))
        grid[2].imshow(psf[:, ny // 2, :].T, **dkwargs, extent=(*kz, *ky))
        grid[1].imshow(psf[:, :, nx // 2], **dkwargs, extent=(*kx, *kz))
    grid[0].axis("off")

    fd = {"fontweight": "bold"}
    # add titles
    grid[3].set_title("$XY$", fd)
    grid[2].set_title("$YZ$", fd)
    grid[1].set_title("$XZ$", fd)
    # remove ticks
    for g in grid:
        g.xaxis.set_major_locator(plt.NullLocator())
        g.yaxis.set_major_locator(plt.NullLocator())
    # add scalebar
    add_scalebar(grid[3], 1, 1, None)
    # return fig and axes
    return fig, grid


def otf_plot(
    otf,
    na=0.85,
    wl=0.52,
    nobj=1.0,
    nsample=None,
    zstep=0.25,
    pixel_size=0.13,
    fig=None,
    loc=111,
    **kwargs,
):
    """"""
    if nsample is None:
        nsample = nobj
    # expand z step
    zstep *= z_squeeze(nobj, nsample, na)
    # update our default kwargs for plotting
    dkwargs = dict(
        norm=mpl.colors.LogNorm(vmin=kwargs.pop("vmin", None), vmax=kwargs.pop("vmax", None)),
        interpolation="nearest",
        cmap="inferno",
    )
    dkwargs.update(kwargs)
    # make the fig if one isn't passed
    if fig is None:
        fig = plt.figure(None, (8.0, 8.0))

    grid = mpl.ImageGrid(fig, loc, nrows_ncols=(2, 2), axes_pad=0.3)

    nz, ny, nx = otf.shape
    assert nx == ny
    kz, ky, kx = [fft_max_min(n, d) for n, d in zip(otf.shape, (zstep, pixel_size, pixel_size))]

    grid[3].imshow(otf[nz // 2, :, :], **dkwargs, extent=(*kx, *ky))
    grid[2].imshow(otf[:, ny // 2, :].T, **dkwargs, extent=(*kz, *ky))
    grid[1].imshow(otf[:, :, nx // 2], **dkwargs, extent=(*kx, *kz))
    grid[0].axis("off")

    fd = {"fontweight": "bold"}
    grid[3].set_title("$k_{XY}$", fd)
    grid[2].set_title("$k_{YZ}$", fd)
    grid[1].set_title("$k_{XZ}$", fd)

    for g in grid:
        g.xaxis.set_major_locator(plt.NullLocator())
        g.yaxis.set_major_locator(plt.NullLocator())

    # calculate the angle of the marginal rays
    a = np.arcsin(min(1, na / nsample))
    # make a circle of the OTF limits
    c = patches.Circle((0, 0), 2 * na / wl, ec="w", lw=2, fill=None)
    grid[3].add_patch(c)
    # add bowties
    n_l = nsample / wl
    for b, g in zip((0, np.pi / 2), grid[1:3]):
        for j in (0, np.pi):
            for i in (0, np.pi):
                c2 = patches.Wedge(
                    (n_l * np.sin(a + b + j), n_l * np.cos(a + b + i)),
                    n_l,
                    np.rad2deg(-a - np.pi / 2 + i * np.cos(b) - (j + np.pi) * np.sin(b) + b),
                    np.rad2deg(a - np.pi / 2 + i * np.cos(b) - (j + np.pi) * np.sin(b) + b),
                    width=0,
                    ec="w",
                    lw=1,
                    fill=None,
                )
                g.add_patch(c2)
    # add scalebar
    add_scalebar(grid[3], 1, 1, None)

    return fig, grid
