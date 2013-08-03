import numpy as np
import quantities as q
from unittest import TestCase
from syris import config as cfg
from syris.opticalelements.materials import PMASFMaterial
import os


def pmasf_required(func):
    func.pmasf_required = 1
    return func


class TestPMASFMaterial(TestCase):

    def setUp(self):
        if not os.path.exists(cfg.PMASF_FILE):
            # Remote access.
            cfg.PMASF_FILE = "ssh hopped_ufo /home/farago/software/asf/pmasf"

    @pmasf_required
    def test_one_energy(self):
        material = PMASFMaterial("PMMA", [20] * q.keV)
        self.assertEqual(len(material.refractive_indices), 1)

    @pmasf_required
    def test_multiple_energies(self):
        energies = np.linspace(15, 25, 10) * q.keV
        material = PMASFMaterial("PMMA", energies)
        self.assertEqual(len(material.refractive_indices), 10)

    @pmasf_required
    def test_wrong_energy(self):
        self.assertRaises(RuntimeError, PMASFMaterial, "PMMA", [0] * q.keV)

    @pmasf_required
    def test_wrong_material(self):
        self.assertRaises(RuntimeError, PMASFMaterial, "asd", [0] * q.keV)
