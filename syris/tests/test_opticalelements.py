import numpy as np
import quantities as q
import syris
from syris.graphicalobjects import SimpleGraphicalObject
from syris.materials import Material
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength, transfer
from syris.tests import SyrisTest, slow


@slow
class TestOpticalElement(SyrisTest):

    def setUp(self):
        syris.init()
        self.go = SimpleGraphicalObject(np.arange(4 ** 2).reshape(4, 4) * q.um)
        energies = range(10, 20) * q.keV
        self.energy = energies[len(energies) / 2]
        self.material = Material('foo', np.arange(len(energies), dtype=np.complex), energies)
        self.elem = OpticalElement(self.go, self.material)

    def test_transfer(self):
        transferred = self.elem.transfer(self.energy).get()
        gt = transfer(self.go.thickness, self.material.get_refractive_index(self.energy),
                      energy_to_wavelength(self.energy)).get()
        np.testing.assert_almost_equal(gt, transferred)
