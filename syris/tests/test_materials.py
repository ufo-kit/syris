import numpy as np
import quantities as q
import syris
from distutils.spawn import find_executable
from syris import config as cfg
from syris.materials import Material, MaterialError, make_pmasf, make_henke
import os
from syris.tests import SyrisTest, slow


class TestMaterial(SyrisTest):

    def setUp(self):
        syris.init()
        self.energies = np.arange(1, 5, 1) * q.keV
        self.refractive_indices = np.array([i + i * 1j for i in range(1, len(self.energies) + 1)])

    def test_comparison(self):
        m_0 = Material("PMMA", self.refractive_indices, self.energies)
        m_1 = Material("glass", self.refractive_indices, self.energies)
        m_2 = Material("PMMA", self.refractive_indices, self.energies)

        self.assertEqual(m_0, m_2)
        self.assertNotEqual(m_0, m_1)
        self.assertNotEqual(m_0, None)
        self.assertNotEqual(m_0, 1)

    def test_hashing(self):
        m_0 = Material("PMMA", self.refractive_indices, self.energies)
        m_1 = Material("glass", self.refractive_indices, self.energies)
        m_2 = Material("PMMA", self.refractive_indices, self.energies)

        self.assertEqual(len(set([m_1, m_0, m_1, m_2])), 2)

    def test_interpolation(self):
        mat = Material('foo', self.refractive_indices, self.energies)
        index = mat.get_refractive_index(2400 * q.eV)
        self.assertAlmostEqual(2.4, index.real, places=5)
        self.assertAlmostEqual(2.4, index.imag, places=5)

    def test_interpolation_out_of_bounds(self):
        mat = Material('foo', self.refractive_indices, self.energies)
        self.assertRaises(ValueError, mat.get_refractive_index, 1 * q.eV)
        self.assertRaises(ValueError, mat.get_refractive_index, 1 * q.MeV)

    def test_interpolation_few_points(self):
        energies = [0] * q.eV
        indices = [1 + 1j]
        self.assertRaises(MaterialError, Material, 'foo', indices, energies)


@slow
class TestPMASFMaterial(SyrisTest):

    def setUp(self):
        syris.init()
        self.energies = np.arange(1, 5, 1) * q.keV
        self.refractive_indices = np.array([i + i * 1j for i in range(1, len(self.energies) + 1)])

    def test_multiple_energies(self):
        if find_executable(cfg.PMASF_FILE):
            energies = np.linspace(15, 25, 10) * q.keV
            material = make_pmasf("PMMA", energies)
            self.assertEqual(len(material.refractive_indices), 10)

    def test_wrong_energy(self):
        if find_executable(cfg.PMASF_FILE):
            self.assertRaises(RuntimeError, make_pmasf, "PMMA", [0] * q.keV)

    def test_wrong_material(self):
        if find_executable(cfg.PMASF_FILE):
            self.assertRaises(RuntimeError, make_pmasf, "asd", [0] * q.keV)

    def test_wrong_executable(self):
        cfg.PMASF_FILE = 'dskjfhjsaf'
        self.assertRaises(RuntimeError, make_pmasf, "PMMA", [0] * q.keV)


@slow
class TestHenkeMaterial(SyrisTest):

    def test_creation(self):
        syris.init()
        energies = np.arange(1, 10, 1) * q.keV
        make_henke('foo', energies, formula='H')

    def test_out_of_range(self):
        # Minimum too small
        energies = np.arange(1, 1000, 1) * q.eV
        self.assertRaises(ValueError, make_henke, 'foo', energies, formula='H')

        # Maximum too big
        energies = np.arange(1, 1000, 1) * q.keV
        self.assertRaises(ValueError, make_henke, 'foo', energies, formula='H')

    def test_wrong_formula(self):
        energies = np.arange(1, 10, 1) * q.keV
        self.assertRaises(MaterialError, make_henke, 'foo', energies, formula='xxx')
