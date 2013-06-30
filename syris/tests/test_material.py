import numpy as np
import quantities as q
from unittest import TestCase
from syris import config as cfg
from syris.opticalelements.material import PMASFMaterial
from testfixtures import ShouldRaise

cfg.PMASF_FILE = "/home/farago/software/asf/pmasf"


class TestPMASFMaterial(TestCase):

    def test_one_energy(self):
        material = PMASFMaterial("PMMA", [20] * q.keV)
        self.assertEqual(len(material.refractive_indices), 1)

    def test_multiple_energies(self):
        energies = np.linspace(15, 25, 10) * q.keV
        material = PMASFMaterial("PMMA", energies)
        self.assertEqual(len(material.refractive_indices), 10)

    def test_wrong_energy(self):
        with ShouldRaise(RuntimeError):
            PMASFMaterial("PMMA", [0] * q.keV)

    def test_wrong_material(self):
        with ShouldRaise(RuntimeError):
            PMASFMaterial("asd", [0] * q.keV)
