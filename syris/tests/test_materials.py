import numpy as np
import quantities as q
from syris import config as cfg
from syris.opticalelements.materials import PMASFMaterial, Material, HenkeMaterial, MaterialError
import os
from syris.tests.base import SyrisTest


def pmasf_required(func):
    func.pmasf_required = 1
    return func


class TestPMASFMaterial(SyrisTest):

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

    def test_comparison(self):
        m_0 = Material("PMMA", None)
        m_1 = Material("glass", None)
        m_2 = Material("PMMA", None)

        self.assertEqual(m_0, m_2)
        self.assertNotEqual(m_0, m_1)
        self.assertNotEqual(m_0, None)
        self.assertNotEqual(m_0, 1)

    def test_hashing(self):
        m_0 = Material("PMMA", None)
        m_1 = Material("glass", None)
        m_2 = Material("PMMA", None)

        self.assertEqual(len(set([m_1, m_0, m_1, m_2])), 2)


class TestHenkeMaterial(SyrisTest):

    def test_creation(self):
        energies = [100, 1000] * q.eV
        HenkeMaterial('foo', energies, formula='H')

    def test_out_of_range(self):
        # Minimum too small
        energies = [10, 1000] * q.eV
        self.assertRaises(ValueError, HenkeMaterial, 'foo', energies, formula='H')

        # Maximum too big
        energies = [100, 1e7] * q.eV
        self.assertRaises(ValueError, HenkeMaterial, 'foo', energies, formula='H')

    def test_wrong_formula(self):
        energies = [100, 1000] * q.eV
        self.assertRaises(MaterialError, HenkeMaterial, 'foo', energies, formula='xxx')
