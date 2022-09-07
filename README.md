[![PyPI version](https://badge.fury.io/py/pyotf.svg)](https://badge.fury.io/py/pyotf)
[![PyPI downloads](https://img.shields.io/pypi/dm/pyotf.svg?color=magenta&logo=pypi)](https://pypi.org/project/pyotf)
[![Anaconda version](https://anaconda.org/david-hoffman/pyotf/badges/downloads.svg)](https://anaconda.org/david-hoffman/pyotf)
[![Conda Badge](https://anaconda.org/david-hoffman/pyotf/badges/installer/conda.svg)](https://anaconda.org/david-hoffman/pyotf)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![ci](https://github.com/david-hoffman/pyOTF/workflows/ci/badge.svg)](https://github.com/david-hoffman/pyOTF/actions?query=workflow%3Aci)
[![Create Release](https://github.com/david-hoffman/pyOTF/workflows/Create%20Release/badge.svg)](https://github.com/david-hoffman/pyOTF/actions?query=workflow%3A%22Create+Release%22)

# pyotf

A simulation software package for modelling optical transfer functions (OTF)/point spread functions (PSF) of optical microscopes written in python.

## Introduction

The majority of this package's documentation is included in the source code and should be available in any interactive session. The intent of this document is to give a quick overview of the package's features and potential uses. Much of the code has been designed with interactive sessions in mind but it should still be usable in larger scripts and programs.

## Installation

Installation is simplest with `conda` or `pip`:

```bash
conda install -c david-hoffman pyotf
```

```bash
pip install pyotf
```

## Components

The package is made up of four component modules:

- `otf.py` which contains classes for generating different types of OTFs and PSFs
- `phase_retrieval.py` which contains functions and classes to perform iterative [phase retrieval][3] of the rear aperature of the optical system
- `zernike.py` which contains functions for calculating [Zernike Polynomials](https://en.wikipedia.org/wiki/Zernike_polynomials)
- `utils.py` which contains various utility functions used throughout the package.

### otf.py

![Comparison of HanserPSF and SheppardPSF Outputs](https://github.com/david-hoffman/pyOTF/blob/master/fixtures/otf.png?raw=true "Output of python -m pyotf.otf")

Two models of optical imaging systems are available in this module one described by [Hanser et al][1] and one described by [Arnison and Sheppard][2]. They are, in fact, mathematically equivalent but in practice each have their strengths and weaknesses. A big benefit of `HanserPSF` is that it allows one to calculate selected z planes of the PSF. However, if the choosen z-planes are not equispaced then the field OTF (`OTFa`) and intensity OTF (`OTFi`) calculated from the model won't make physical sense.

Both the `SheppardPSF` and `HanserPSF` have much the same interface. When instantiating them the user must provide a set of model parameters. To fully describe a PSF or OTF of an objective lens, assuming no abberation, we generally need a few parameters:

- The wavelength of operation (assume monochromatic light)
- the numerical aperature of the objective
- the index of refraction of the medium

For numerical calculations we'll also want to know the x/y resolution and number of points. Note that it is assumed that z is the optical axis of the objective lens.

### phaseretrieval.py

The phase retrieval algorithm implemented in this module is described by [Hanser et. al][3].

An example for how to use these functions can be found at the end of the file:

```python
    # phase retrieve a pupil
    from pathlib import Path
    import time
    import warnings
    import tifffile as tif
    from .utils import prep_data_for_PR

    # read in data from fixtures
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = tif.imread(
            str(
                Path(__file__).parent.parent / "fixtures/psf_wl520nm_z300nm_x130nm_na0.85_n1.0.tif"
            )
        )
        # prep data
    data_prepped = prep_data_for_PR(data, 256, 1.1)

    # set up model params
    params = dict(wl=520, na=0.85, ni=1.0, res=130, zres=300)

    # retrieve the phase
    pr_start = time.time()
    print("Starting phase retrieval ... ", end="", flush=True)
    pr_result = retrieve_phase(data_prepped, params, 100, 1e-4, 1e-4)
    pr_time = time.time() - pr_start
    print(f"{pr_time:.1f} seconds were required to retrieve the pupil function")

    # plot
    pr_result.plot()
    pr_result.plot_convergence()

    # fit to zernikes
    zd_start = time.time()
    print("Starting zernike decomposition ... ", end="", flush=True)
    pr_result.fit_to_zernikes(120)
    pr_result.zd_result.plot()
    zd_time = time.time() - zd_start
    print(f"{zd_time:.1f} seconds were required to fit 120 Zernikes")

    # plot
    pr_result.zd_result.plot_named_coefs()
    pr_result.zd_result.plot_coefs()

    # show
    plt.show()
```

Below is a plot of the phase and magnitude of the retrieved pupil function from a PSF recorded from [this](https://science.sciencemag.org/content/367/6475/eaaz5357) instrument. To generate this plot we simply call the `plot` method of the `PhaseRetrievalResult` object (in this case `pr_result`).

![ ](https://github.com/david-hoffman/pyOTF/blob/master/fixtures/PR%20Result.png?raw=true "The phase and magnitude of the retrieved pupil function")

And here the phase and magnitude have been fitted to 120 zernike polynomials. To generate this plot we simply call the `plot` method of the `ZernikeDecomposition` object (in this case `pr_result.zd_result`).

![ ](https://github.com/david-hoffman/pyOTF/blob/master/fixtures/PR%20Result%20ZD.png?raw=true "The phase and magnitude decomposed into 120 zernike polynomials")

We can plot the magnitude of the first 15 named phase coefficients by calling `pr_result.zd_result.plot_named_coefs()`.

![ ](https://github.com/david-hoffman/pyOTF/blob/master/fixtures/Named%20Coefs.png?raw=true "The first 15 zernike polynomial coefficients which correspond to named aberrations.")

**NOTE:** If all that is needed is phase, e.g. for adaptive optical correction, then most normal ways of estimating the background should be sufficient and you can use the `phase_only` keyword. However, if you want to properly model your PSF for something like deconvolution then you should be aware that the magnitude estimate is _incredibly_ sensitive to the background correction applied to the data prior to running the algorithm, and multiple background methods/parameters should be tried.

### zernike.py

![First 15 zernike Polynomials plotted on the unit disk](https://github.com/david-hoffman/pyOTF/blob/master/fixtures/zernike.png?raw=true "Output of python -m pyotf.zernike")

![Corresponding aberrations](https://github.com/david-hoffman/pyOTF/blob/master/fixtures/aberrations.png?raw=true)

[Zernike Polynomials](https://en.wikipedia.org/wiki/Zernike_polynomials) are orthonormal functions defined over the unit disk. Being orthonormal any function defined on a unit disk has a unique decomposition into Zernike polynomials. In this package, the Zernike polynomials are used to quantify the abberation of the phase and magnitude of the retrieved back pupil of the optical system. To do so one can call the `fit_to_zernikes` method of a `PhaseRetrievalResult` object, which will fit a specified number of Zernike modes to the back pupil's retrieved phase and magnitude, each independently, and return a `ZernikeDecomposition` object. The `ZernikeDecomposition` that is returned is also saved as an attribute of the `PhaseRetreivalResult` object on which the `fit_to_zernikes` method was called, for convenience. `ZernikeDecomposition` objects have plotting methods so that the user can inspect the decomposition. `ZernikeDecomposition` objects also have methods for reconstructing the phase, magnitude or complete complex pupil which can be fed back into `HanserPSF` to generate an abberated, but noise free, PSF. The method for doing this simply is currently a member of the `PhaseRetreivalResult` class but will probably be moved to the `ZernikeDecomposition` class later.

### utils.py

Most of the contents of `utils` won't be useful to the average user save one function: `prep_data_for_PR(data, xysize=None, multiplier=1.5)`. `prep_data_for_PR` can, as its name suggests, be used to quickly prep PSF image data for phase retrieval using the `retrieve_phase` function of the `phase_retrieval` module.

## LabVIEW API

An example of inputing a 3D stack and running this python function from LabVIEW (>2018) is given in `\labview\Test Phase Retrieval.vi`

### References

1. [Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems.Optics Letters 2003, 28 (10), 801.][1]

2. [Arnison, M. R.; Sheppard, C. J. R. A 3D Vectorial Optical Transfer Function Suitable for Arbitrary Pupil Functions. Optics Communications 2002, 211 (1–6), 53–63.][2]

3. [Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase-Retrieved Pupil Functions in Wide-Field Fluorescence Microscopy. Journal of Microscopy 2004, 216 (1), 32–48.][3]

[1]: http://dx.doi.org/10.1364/OL.28.000801, "Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems.Optics Letters 2003, 28 (10), 801."

[2]: http://dx.doi.org/10.1016/S0030-4018(02)01857-6 "Arnison, M. R.; Sheppard, C. J. R. A 3D Vectorial Optical Transfer Function Suitable for Arbitrary Pupil Functions. Optics Communications 2002, 211 (1–6), 53–63."

[3]: https://doi.org/10.1111/j.0022-2720.2004.01393.x "Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase-Retrieved Pupil Functions in Wide-Field Fluorescence Microscopy. Journal of Microscopy 2004, 216 (1), 32–48."
