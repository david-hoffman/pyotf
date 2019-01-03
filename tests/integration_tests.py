#!/usr/bin/env python
# -*- coding: utf-8 -*-
# integration_tests.py
"""
Small test suite

Copyright David Hoffman, 2018
"""

from nose.tools import *
import unittest
from pyOTF.otf import *
from pyOTF.phaseretrieval import *
import numpy as np


class TestHanserPhaseRetrieval(unittest.TestCase):
    """Test for self consistency, generate a pupil with random zernike
    coefficients generate a psf and phase retrieve it."""

    def setUp(self):
        """Set up the test"""
        # model kwargs
        self.model_kwargs = dict(
            wl=525,
            na=0.85,
            ni=1.0,
            res=140,
            size=64,
            zrange=[-1000, 0, 1000, 3000],
            vec_corr="none",
            condition="none"
        )
        # make the model
        model = HanserPSF(**self.model_kwargs)
        # extract kr
        model._gen_kr()
        kr = model._kr
        theta = model._phi
        # make zernikes (need to convert kr to r where r = 1 when kr is at
        # diffraction limit)
        r = kr * model.wl / model.na
        zerns = zernike(r, theta, np.arange(5, 16))
        # make fack phase and magnitude coefs
        self.pcoefs = np.random.rand(zerns.shape[0])
        self.mcoefs = np.random.rand(zerns.shape[0])
        self.pupil_phase = (zerns * self.pcoefs[:, np.newaxis, np.newaxis]).sum(0)
        self.pupil_mag = (zerns * self.mcoefs[:, np.newaxis, np.newaxis]).sum(0)
        self.pupil_mag = self.pupil_mag + model._gen_pupil() * (2.0 - self.pupil_mag.min())
        # phase only test
        model._gen_psf(self.pupil_mag * np.exp(1j * self.pupil_phase) * model._gen_pupil())
        self.PSFi = model.PSFi
        # we have to converge really close for this to work.
        self.PR_result = retrieve_phase(self.PSFi, self.model_kwargs,
                                   max_iters=200, pupil_tol=0,
                                   mse_tol=0, phase_only=False)

    def test_phase_and_mag(self):
        """Make sure phase retrieval returns same phase and magnitude"""
        np.testing.assert_allclose(fftshift(self.pupil_mag),
                                   self.PR_result.mag, err_msg="Mag failed")
        np.testing.assert_allclose(fftshift(self.pupil_phase),
                                   self.PR_result.phase, err_msg="Phase failed")

    def test_zernike_modes(self):
        """Make sure the fitted zernike modes agree"""
        self.PR_result.fit_to_zernikes(15)
        np.testing.assert_allclose(self.PR_result.zd_result.pcoefs[4:],
                                   self.pcoefs, err_msg="Phase coefs failed")
        np.testing.assert_allclose(self.PR_result.zd_result.mcoefs[4:],
                                   self.mcoefs, err_msg="Mag coefs failed")

    def test_psf_mse(self):
        """Does the phase retrieved PSF converge to the fake PSF"""
        np.testing.assert_allclose(self.PR_result.model.PSFi, self.PSFi)
