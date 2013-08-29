import quantities as q
from syris.devices.lenses import Lens
from syris.tests.base import SyrisTest


class TestLenses(SyrisTest):

    def setUp(self):
        self.f_length = 100 * q.mm
        self.f_number = 1.4
        self.magnification = 1
        self.transmission_eff = 0.5
        self.psf_sigma = 1 * q.um, 1 * q.um
        self.lens = Lens(self.f_number, self.f_length, self.magnification,
                         self.transmission_eff, self.psf_sigma)

    def test_init(self):
        self.assertRaises(ValueError, Lens, self.f_number, self.f_length,
                          self.magnification, -1, self.psf_sigma)
        self.assertRaises(ValueError, Lens, self.f_number, self.f_length,
                          self.magnification, 10, self.psf_sigma)

    def test_numerical_aperture(self):
        self.assertAlmostEquals(self.lens.numerical_aperture, 0.17579063848)
        self.lens.magnification = 3
        self.assertAlmostEquals(self.lens.numerical_aperture, 0.25873608522)
