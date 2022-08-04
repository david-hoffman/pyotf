#!/usr/bin/env python
# -*- coding: utf-8 -*-
# display.py
"""
Display functions for PSFs and OTFs.

Copyright (c) 2021, David Hoffman
"""

import typing

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def max_min(n: int, d: float) -> typing.Tuple[float, float]:
    """Return max and min extents.

    Parameters
    ----------
    n : int
        Number of pixels
    d : float
        Pixel pitch / spacing

    Returns
    -------
    min, max : floats
        Min and max extents

    >>> max_min(11, 1.0)
    (-5.0, 5.0)

    >>> max_min(10, 1.0)
    (-5.0, 4.0)
    """
    min_ = (n // 2) * d
    max_ = (n // 2 - (n - 1) % 2) * d
    return -min_, max_


def fft_max_min(n: int, d: float) -> typing.Tuple[float, float]:
    """Return max and min Fourier extents.

    Parameters
    ----------
    n : int
        Number of pixels
    d : float
        Pixel pitch / spacing

    Returns
    -------
    min, max : floats
        Min and max extents

    >>> fft_max_min(10, 1.0)
    (-0.5, 0.4)

    >>> fft_max_min(5, 1 / 3)
    (-1.2, 1.2)
    """
    step_size = 1 / d / n
    return max_min(n, step_size)


def z_squeeze(n1: float, n2: float, na: float) -> float:
    """Calculate amount z expands or contracts.

    When using an objective designed for one index (n1) to image into a medium with another
    index (n2) the amount the focal plane moves within the sample will be different from the
    amount the objective moves relative to the sample.
    """
    if n1 == n2:
        return 1

    def func(n):
        return n - np.sqrt(max(0, n**2 - na**2))

    return func(n1) / func(n2)


def psf_plot(
    psf: np.ndarray,
    *,
    zres: float,
    res: float,
    fig: mpl.figure.Figure = None,
    loc: int = 111,
    mip: bool = True,
    **kwargs,
):
    """Make a nice plot for a light microscopy point spread function (PSF).

    Parameters
    ----------
    psf : nd.array (3d)
        A 3d array representing an intensity PSF of a microscope.
        NOTE: assumes that the axial steps are regularly spaced!
    zres : float
        Axial step/pixel size
    res : float
        Lateral pixel size
    fig : Figure (optional)
        Figure in which to place the plot
    loc : int (optional)
        Location to place the plot in the figure (see ImageGrid docs for details)
    mip : bool (optional)
        Choose to display as a maximum intensity projection or central slices
        NOTE: data  needs to be centered for the slice plot to work.
    kwargs : optional keyword arguments
        kwargs are passed along to the `imshow` calls

    Returns
    -------
    fig : Figure
    axs : Axes

    Notes
    -----
    This function is NOT unit aware, so make sure the units for `zres` and `res` are the same.
    """
    # update our default kwargs for plotting
    dkwargs = dict(
        norm=mpl.colors.PowerNorm(
            gamma=kwargs.pop("gamma", 0.5),
            vmin=kwargs.pop("vmin", None),
            vmax=kwargs.pop("vmax", None),
        ),
        interpolation="nearest",
        cmap="inferno",
    )
    dkwargs.update(kwargs)
    # make the fig if one isn't passed
    if fig is None:
        fig = plt.figure(None, (8.0, 8.0))

    grid = ImageGrid(fig, loc, nrows_ncols=(2, 2), axes_pad=0.3)
    # calc extents
    nz, ny, nx = psf.shape
    kz, ky, kx = [max_min(n, d) for n, d in zip(psf.shape, (zres, res, res))]

    # do plotting
    if mip:
        grid[3].imshow(psf.max(0), **dkwargs, extent=(*kx, *ky))
        grid[2].imshow(psf.max(2).T, **dkwargs, extent=(*kz, *ky))
        grid[1].imshow(psf.max(1), **dkwargs, extent=(*kx, *kz))
    else:
        grid[3].imshow(psf[nz // 2, :, :], **dkwargs, extent=(*kx, *ky))
        grid[2].imshow(psf[:, :, nx // 2].T, **dkwargs, extent=(*kz, *ky))
        grid[1].imshow(psf[:, ny // 2, :], **dkwargs, extent=(*kx, *kz))
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
    # return fig and axes
    return fig, grid


def otf_plot(
    otf: np.ndarray,
    *,
    na: float,
    ni: float,
    wl: float,
    zres: float,
    res: float,
    fig: mpl.figure.Figure = None,
    loc: int = 111,
    **kwargs,
):
    """Make a nice plot for a light microscopy optical transfer function (OTF).

    Parameters
    ----------
    otf : nd.array (3d)
        A 3d array representing an intensity OTF of a microscope.
        NOTE: assumes that the axial steps are regularly spaced!
    na : float
        Numerical aperture of the microscope.
    ni : float
        Index of refraction of the medium of measurement
        NOTE: assumes that the objective is used in its intended medium (index matching is satisfied)
    wl : float
        Approximate emission wavelength of the data
    zres : float
        Real space axial step/pixel size (Fourier space units are automatically calculated)
    res : float
        Real space lateral pixel size (Fourier space units are automatically calculated)
    fig : Figure (optional)
        Figure in which to place the plot
    loc : int (optional)
        Location to place the plot in the figure (see ImageGrid docs for details)
    mip : bool (optional)
        Choose to display as a maximum intensity projection or central slices
        NOTE: data  needs to be centered for the slice plot to work.
    kwargs : optional keyword arguments
        kwargs are passed along to the `imshow` calls

    Returns
    -------
    fig : Figure
    axs : Axes

    Notes
    -----
    This function is NOT unit aware, so make sure the units for `zres`, `res`, and `wl` are the same.
    """
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

    grid = ImageGrid(fig, loc, nrows_ncols=(2, 2), axes_pad=0.3)

    nz, ny, nx = otf.shape
    assert nx == ny
    kz, ky, kx = [fft_max_min(n, d) for n, d in zip(otf.shape, (zres, res, res))]

    grid[3].imshow(otf[nz // 2, :, :], **dkwargs, extent=(*kx, *ky))
    grid[2].imshow(otf[:, :, nx // 2].T, **dkwargs, extent=(*kz, *ky))
    grid[1].imshow(otf[:, ny // 2, :], **dkwargs, extent=(*kx, *kz))
    grid[0].axis("off")

    fd = {"fontweight": "bold"}
    grid[3].set_title("$k_{XY}$", fd)
    grid[2].set_title("$k_{YZ}$", fd)
    grid[1].set_title("$k_{XZ}$", fd)

    for g in grid:
        g.xaxis.set_major_locator(plt.NullLocator())
        g.yaxis.set_major_locator(plt.NullLocator())

    # calculate the angle of the marginal rays
    a = np.arcsin(min(1, na / ni))
    # make a circle of the OTF limits
    c = patches.Circle((0, 0), 2 * na / wl, ec="w", lw=2, fill=None)
    grid[3].add_patch(c)
    # add bowties
    n_l = ni / wl
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

    return fig, grid
