Syris
=====

.. image:: https://img.shields.io/badge/Python-3.6+-blue
    :target: https://www.python.org/downloads/release/python-360

.. image:: https://travis-ci.com/ufo-kit/syris.svg?branch=master
    :target: https://travis-ci.com/ufo-kit/syris

.. image:: https://readthedocs.org/projects/syris/badge/?version=latest
    :target: http://syris.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/ufo-kit/syris/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ufo-kit/syris

*Syris* (**sy**\ nchrotron **r**\ adiation **i**\ maging **s**\ imulation) is a
framework for simulations of X-ray absorption and phase contrast dynamic imaging
experiments, like time-resolved radiography, tomography or laminography. It
includes X-ray sources, various sample shape creation possibilities, complex
refractive index lookup options, motion model and indirect detection model
(scintillator combined with a conventional camera). Phase contrast is simulated
by the Angular spectrum method, which enables one to include various optical
elements in the simulation, e.g. gratings and X-ray lenses.

Compute-intensive algorithms like Fourier transforms, sample shape creation and
free-space propagation are implemented by using OpenCL, which enables one to
execute the code on graphic cards.

There are numerous examples of how to use *syris* described below which ship
directly with the code. Enjoy!


Usage
-----

The first thing you have to do is to initialize *syris* by the ``syris.init()``
function. After that you only need to do whatever is necessary for your program.
A simple white beam propagation example looks like this:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import quantities as q
    import syris
    from syris.physics import propagate
    from syris.bodies.simple import make_sphere
    from syris.materials import make_henke

    syris.init()
    energies = np.arange(10, 30) * q.keV
    n = 1024
    pixel_size = 0.4 * q.um
    distance = 2 * q.m
    material = make_henke('PMMA', energies)

    sample = make_sphere(n, n / 4 * pixel_size, pixel_size, material=material)
    image = propagate([sample], (n, n), energies, distance, pixel_size).get()
    plt.imshow(image)
    plt.show()

For more detailed information please see the `reference <https://syris.readthedocs.io/en/latest>`_.


Examples
********

Are stored in the **examples** directory. Make sure you install their
dependencies by ``pip install -r requirements-examples.txt`` and then you can
run them from the top-level directory of syris repository by e.g. ``python -m
examples.simple``.


Citation
--------

If you use this software for publishing your data, we kindly ask to cite the article below.

Faragó, T., Mikulík, P., Ershov, A., Vogelgesang, M., Hänschke, D. & Baumbach,
T. (2017). J. Synchrotron Rad. 24, https://doi.org/10.1107/S1600577517012255
