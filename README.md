# PyRCWA (v0.0.1)
![Read the Docs](https://img.shields.io/readthedocs/pyrcwa)
![GitHub](https://img.shields.io/github/license/vitamingcheng/PyRCWA)
![PyPI](https://img.shields.io/pypi/v/PyRCWA)


The vision of PyRCWA is to be a high-performance numerical simulation tool that can be used to solve electromagnetic field propagation in multi-layered periodic structures with **complex material systems**.

# Getting Started
## Installation
It's recommend to install this software with pip:
```shell
pip install rcwa
```

## Example Program
To see a simple example, run:
```shell
python -m rcwa.examples.bragg_mirror
```

This should run an example with a 10-layer bragg mirror (also known as a dielectric mirror), which can have very high reflectance near its design wavelength, and output the reflectance as a function of wavelength, as seen below:

## a new example
```python
from pyrcwa.source import RCWASource
from pyrcwa.model import RCWALayer, RCWAModel
from pyrcwa.settings import RCWASettings
from pyrcwa.solver import RCWASolver

theSource = RCWASource(wavelength=0.5e-6, alpha=0, theta=0.0, phi=90, phase=0)
theRefLayer = RCWALayer(layer_type="Ref", epsilon=1, mu=1)
theGratingLayer = RCWALayer(
    layer_type="1d grating",
    epsilon1=1.5*1.5,
    epsilon2=1,
    mu1=1,
    mu2=1,
    thickness=0.5e-6,
    duty_cycle=0.5,
    period=1e-6
)
theTrnLayer = RCWALayer(layer_type="Trn", epsilon=1.5*1.5, mu=1)

theModel = RCWAModel(
    layers=[theRefLayer, theGratingLayer, theTrnLayer],
    spatial_period=(theGratingLayer.period, theGratingLayer.period)
)

theSettings = RCWASettings(
    harmonics=(1, 0),
    fft_resolution=(1001, 1001)
)

theRCWASolver = RCWASolver(
    source=theSource,
    model=theModel,
    settings=theSettings
)

theRCWASolver.solve_new()
theRCWASolver.show()
```

## Features
The feature balabala

## Installation

- Requirements
  - Python version 3.8 or higher
  - For GPU acceleration, balabala
- GPU test

# Official documentation
See the [Readthedocs: Official PyRCWA's documentation](https://pyrcwa.readthedocs.io/en/latest/ "official PyRCWA's documentation")

See the [PyPI: PyRCWA's documentation](https://pypi.org/project/PyRCWA/ "pyPI: PyRCWA's documentation")


# License
This project is distributed under the [MIT license](https://mit-license.org/ "MIT license").

# Acknowledgements
This work is based primarily on a set of lectures and associated course material by [Professor Raymond Rumpf](https://raymondrumpf.com/ "Professor Raymond Rumpf") at the University of Texas, El Paso.

# References
[1] Rakić, Aleksandar D., Aleksandra B. Djurišić, Jovan M. Elazar, and Marian L. Majewski. "Optical properties of metallic films for vertical-cavity optoelectronic devices." Applied optics 37, no. 22 (1998): 5271-5283.
