Syris
=====

*syris* (**sy**\ nchrotron **r**\ adiation **i**\ maging **s**\ imulation) is a
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

You can read more about *Concert* in the official `documentation`_.

.. _documentation: https://syris.readthedocs.io/en/latest/


Citation
--------

Faragó, T., Mikulík, P., Ershov, A., Vogelgesang, M., Hänschke, D. & Baumbach,
T. (2017). J. Synchrotron Rad. 24, https://doi.org/10.1107/S1600577517012255
