import numpy as np
import quantities as q
from syris import config as cfg
from syris.opticalelements.materials import Material, MaterialError, make_pmasf, make_henke
import os
from syris.tests import SyrisTest, pmasf_required, slow


class TestPMASFMaterial(SyrisTest):

    def setUp(self):
        if not os.path.exists(cfg.PMASF_FILE):
            # Remote access.
            cfg.PMASF_FILE = "ssh ufo /home/ws/jd2392/software/asf/pmasf"

    @slow
    @pmasf_required
    def test_one_energy(self):
        material = make_pmasf("PMMA", [20] * q.keV)
        self.assertEqual(len(material.refractive_indices), 1)

    @slow
    @pmasf_required
    def test_multiple_energies(self):
        energies = np.linspace(15, 25, 10) * q.keV
        material = make_pmasf("PMMA", energies)
        self.assertEqual(len(material.refractive_indices), 10)

    @slow
    @pmasf_required
    def test_wrong_energy(self):
        self.assertRaises(RuntimeError, make_pmasf, "PMMA", [0] * q.keV)

    @slow
    @pmasf_required
    def test_wrong_material(self):
        self.assertRaises(RuntimeError, make_pmasf, "asd", [0] * q.keV)

    def test_comparison(self):
        m_0 = Material("PMMA", None, None)
        m_1 = Material("glass", None, None)
        m_2 = Material("PMMA", None, None)

        self.assertEqual(m_0, m_2)
        self.assertNotEqual(m_0, m_1)
        self.assertNotEqual(m_0, None)
        self.assertNotEqual(m_0, 1)

    def test_hashing(self):
        m_0 = Material("PMMA", None, None)
        m_1 = Material("glass", None, None)
        m_2 = Material("PMMA", None, None)

        self.assertEqual(len(set([m_1, m_0, m_1, m_2])), 2)


class TestHenkeMaterial(SyrisTest):

    @slow
    def test_creation(self):
        energies = [100, 1000] * q.eV
        make_henke('foo', energies, formula='H')

    @slow
    def test_out_of_range(self):
        # Minimum too small
        energies = [10, 1000] * q.eV
        self.assertRaises(ValueError, make_henke, 'foo', energies, formula='H')

        # Maximum too big
        energies = [100, 1e7] * q.eV
        self.assertRaises(ValueError, make_henke, 'foo', energies, formula='H')

    @slow
    def test_wrong_formula(self):
        energies = [100, 1000] * q.eV
        self.assertRaises(MaterialError, make_henke, 'foo', energies, formula='xxx')
