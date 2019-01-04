# pyOTF

A simulation software package for modelling optical transfer functions (OTF)/point spread functions (PSF) of optical microscopes written in python.

## Introduction

The majority of this package's documentation is included in the source code and should be available in any interactive session. The intent of this document is to give a quick overview of the package's features and potential uses. Much of the code has been designed with interactive sessions in mind but it should still be usable in larger scripts and programs.

## Installation

The software is very much in alpha phase and installation is simply cloning the repository and adding the folder to your python path. (also at the moment requires [dphutils package](https://github.com/david-hoffman/dphutils))

## Components

The package is made up of four component modules:
- `otf.py` which contains classes for generating different types of OTFs and PSFs
- `phase_retrieval.py` which contains functions and classes to perform iterative [phase retrieval][3] of the rear aperature of the optical system
- `zernike.py` which contains functions for calculating [Zernike Polynomials](https://en.wikipedia.org/wiki/Zernike_polynomials)
- `utils.py` which contains various utility functions used throughout the package.

### otf.py

![Comparison of HanserPSF and SheppardPSF Outputs](https://raw.githubusercontent.com/david-hoffman/pyOTF/master/fixtures/otf.png "Output of python -m pyOTF.otf")

Two models of optical imaging systems are available in this module one described by [Hanser et al][1] and one described by [Arnison and Sheppard][2]. They are, in fact, mathematically equivalent but in practice each have their strengths and weaknesses. A big benefit of `HanserPSF` is that it allows one to calculate selected z planes of the PSF. However, if the choosen z-planes are not equispaced then the field OTF (`OTFa`) and intensity OTF (`OTFi`) calculated from the model won't make physical sense.

Both the `SheppardPSF` and `HanserPSF` have much the same interface. When instantiating them the user must provide a set of model parameters. To fully describe a PSF or OTF of an objective lens, assuming no abberation, we generally need a few parameters:

- The wavelength of operation (assume monochromatic light)
- the numerical aperature of the objective
- the index of refraction of the medium

For numerical calculations we'll also want to know the x/y resolution and number of points. Note that it is assumed that z is the optical axis of the objective lens.

### phase_retrieval.py

The phase retrieval algorithm implemented in this module is described by [Hanser et. al][3].

### zernike.py

![First 15 zernike Polynomials plotted on the unit disk](https://raw.githubusercontent.com/david-hoffman/pyOTF/master/fixtures/zernike.png "Output of python -m pyOTF.zernike")

[Zernike Polynomials](https://en.wikipedia.org/wiki/Zernike_polynomials) are orthonormal functions defined over the unit disk. Being orthonormal any function defined on a unit disk has a unique decomposition into Zernike polynomials. In this package, the Zernike polynomials are used to quantify the abberation of the phase and magnitude of the retrieved back pupil of the optical system. To do so one can call the `fit_to_zernikes` method of a `PhaseRetrievalResult` object, which will fit a specified number of Zernike modes to the back pupil's retrieved phase and magnitude, each independently, and return a `ZernikeDecomposition` object. The `ZernikeDecomposition` that is returned is also saved as an attribute of the `PhaseRetreivalResult` object on which the `fit_to_zernikes` method was called, for convenience. `ZernikeDecomposition` objects have plotting methods so that the user can inspect the decomposition. `ZernikeDecomposition` objects also have methods for reconstructing the phase, magnitude or complete complex pupil which can be fed back into `HanserPSF` to generate an abberated, but noise free, PSF. The method for doing this simply is currently a member of the `PhaseRetreivalResult` class but will probably be moved to the `ZernikeDecomposition` class later.

### utils.py

Most of the contents of `utils` won't be useful to the average user save one function: `prep_data_for_PR(data, xysize=None, multiplier=1.5)`. `prep_data_for_PR` can, as its name suggests, be used to quickly prep PSF image data for phase retrieval using the `retrieve_phase` function of the `phase_retrieval` module.

## LabVIEW API

An example of inputing a 3D stack and running this python function from LabVIEW (>2018) is given in `\labview\Test Phase Retrieval.vi`

## Use cases

### References

1. [Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems.Optics Letters 2003, 28 (10), 801.][1]

2. [Arnison, M. R.; Sheppard, C. J. R. A 3D Vectorial Optical Transfer Function Suitable for Arbitrary Pupil Functions. Optics Communications 2002, 211 (1–6), 53–63.][2]

3. [Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems. Optics Letters 2003, 28 (10), 801.][3]

[1]: http://dx.doi.org/10.1364/OL.28.000801, "Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems.Optics Letters 2003, 28 (10), 801."

[2]: http://dx.doi.org/10.1016/S0030-4018(02)01857-6 "Arnison, M. R.; Sheppard, C. J. R. A 3D Vectorial Optical Transfer Function Suitable for Arbitrary Pupil Functions. Optics Communications 2002, 211 (1–6), 53–63."

[3]: http://dx.doi.org/10.1364/OL.28.000801 "Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W. Phase Retrieval for High-Numerical-Aperture Optical Systems. Optics Letters 2003, 28 (10), 801."
