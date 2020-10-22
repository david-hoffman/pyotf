#!/usr/bin/env python
# -*- coding: utf-8 -*-
# labview.py
"""
Thin wrapper so that labview can perform phase retrieval.

Copyright (c) 2018, David Hoffman
"""

from pyotf.phaseretrieval import retrieve_phase
from pyotf.utils import prep_data_for_PR
import numpy as np


def labview(
    data, wl, na, ni, res, zres, max_iters=200, pupil_tol=1e-8, mse_tol=1e-8, phase_only=False
):
    """Generate a PSF object

    Parameters
    ----------
    data : ndarray (3 dim)
        The experimentally measured PSF of a subdiffractive source
    wl : numeric
        Emission wavelength of the simulation
    na : numeric
        Numerical aperature of the simulation
    ni : numeric
        index of refraction for the media
    res : numeric
        x/y resolution of the simulation, must have same units as wl
    zres : numeric
        z resolution of simuation, must have same units a wl
    max_iters : int
        The maximum number of iterations to run, default is 200
    pupil_tol : float
        the tolerance in percent change in change in pupil, default is 1e-8
    mse_tol : float
        the tolerance in percent change for the mean squared error between
        data and simulated data, default is 1e-8
    phase_only : bool
        True means only the phase of the back pupil is retrieved while the
        amplitude is not.
    """

    # Convert from LabVIEW data types to Python types
    data = np.asarray(data)  # LabView converts arrays to lists.  This will convert back to arrays.
    params = dict(
        wl=wl, na=na, ni=ni, res=res, zres=zres
    )  # LabView clusters appear as tuples.  This creates the dict.

    # Preprocess data (background removal, filtering, etc...)
    data_prepped = prep_data_for_PR(data)

    # Phase retrieval
    pr_result = retrieve_phase(data_prepped, params, max_iters, pupil_tol, mse_tol, phase_only)

    return (
        pr_result.phase.tolist()
    )  # LabVIEW wants lists instead of arrays.  This will convert back to lists.


if __name__ == "__main__":
    # phase retrieve a pupil
    import os
    import time
    import logging
    import tifffile as tif
    from matplotlib import pyplot as plt

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    # read in data from fixtures
    data = tif.imread(
        os.path.split(__file__)[0] + "/fixtures/psf_wl520nm_z300nm_x130nm_na0.85_n1.0.tif"
    )
    # set up model params
    params = dict(wl=520, na=0.85, ni=1.0, res=130, zres=300)
    # retrieve the phase
    pr_start = time.time()
    print("Starting phase retrieval with data of size {}".format(data.shape))
    phase = labview(data[6:-5], **params, pupil_tol=1e-6)
    phase = np.asarray(phase)
    print("It took {} seconds to retrieve the pupil function".format(time.time() - pr_start))
    # plot
    max_val = abs(phase).max()
    plt.matshow(phase, cmap="seismic", vmin=-max_val, vmax=max_val)
    plt.colorbar()
    # show
    plt.show()
