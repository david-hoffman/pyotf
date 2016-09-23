import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import ifftshift, fftshift, fftn
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import ifftshift, fftshift, fftn

from .utils import *
from . import HanserPSF
from skimage.restoration import unwrap_phase


class PhaseRetrievalResult(object):
    def __init__(self, mag, phase, mse, pupil_diff, mse_diff, pupils=None):
        self.mag = mag
        self.phase = phase
        self.mse = mse
        self.pupil_diff = pupil_diff
        self.mse_diff = mse_diff
        if pupils is not None:
            self.pupils = pupils


def _retrieve_phase_iter(mag, model, pupil=None):
    """The phase retrieval step"""
    # generate pupil if first iter
    if pupil is None:
        pupil = model._gen_pupil()
    # generate the psf
    model._gen_psf(pupil)
    # keep phase
    phase = np.angle(model.PSFa.squeeze())
    # replace magnitude with experimentally measured mag
    new_psf = mag * np.exp(1j * phase)
    # generate the new pupils
    new_pupils = fftn(fftshift(new_psf, axes=(1, 2)), axes=(1, 2))
    # undo defocus and take the mean and mask off values outside passband
    # might want to unwrap phase before taking mean
    new_pupil = model._gen_pupil() * (new_pupils / model._calc_defocus()).mean(0)
    # if phase only discard magnitude info, the following avoids divide by zero
    return new_pupil, new_pupils


def _calc_mse(data1, data2):
    """utility to calculate mean square error"""
    return ((data1 - data2) ** 2).mean()


def retrieve_phase(data, params, max_iters=200, pupil_tol=1e-10,
                   mse_tol=np.nan, phase_only=False):
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
    # assume that data prep has been handled outside function
    # The field magnitude is the square root of the intensity
    mag = psqrt(data)
    # generate a model from parameters
    model = HanserPSF(**params)
    # make the first mean squared error between data and model
    old_mse = _calc_mse(data, model.PSFi)
    # start a list for iteration
    mse = [old_mse]
    mse_diff = np.zeros(max_iters)
    pupil_diff = np.zeros(max_iters)
    # generate a pupil to start with
    old_pupil = model._gen_pupil()
    # save it as a mask
    mask = old_pupil.real
    # iterate
    for i in range(max_iters):
        # retrieve new pupil
        new_pupil, pupils = _retrieve_phase_iter(mag, model, old_pupil)
        # if phase only discard magnitude info
        if phase_only:
            new_pupil = np.exp(1j * np.angle(new_pupil)) * mask
        # generate new mse and add it to the list
        new_mse = _calc_mse(data, model.PSFi)
        mse.append(new_mse)
        # calculate the difference in mse to test for convergence
        mse_diff[i] = abs(old_mse - new_mse) / old_mse
        # update old_mse
        old_mse = new_mse
        # calculate the difference in pupil
        pupil_diff[i] = (abs(old_pupil - new_pupil)**2).mean() / (abs(old_pupil)**2).mean()
        old_pupil = new_pupil
        # check tolerances
        if pupil_diff[i] < pupil_tol or mse_diff[i] < mse_tol:
            break
    pupil_diff = pupil_diff[:i + 1]
    mse_diff = mse_diff[:i + 1]
    mask = ifftshift((model._gen_pupil().real))
    phase = unwrap_phase(ifftshift(np.angle(old_pupil))) * mask
    magnitude = ifftshift(abs(old_pupil)) * mask
    return PhaseRetrievalResult(magnitude, phase, np.array(mse), pupil_diff, mse_diff)
