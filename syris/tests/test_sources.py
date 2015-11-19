import numpy as np
import quantities as q
import syris
from syris.devices.sources import BendingMagnet
from syris.geometry import Trajectory
from syris.tests import SyrisTest, slow


class TestSources(SyrisTest):

    def setUp(self):
        syris.init()
        self.energies = np.arange(14.8, 15, 0.1) * q.keV
        self.angle_step = 10
        self.ps = 10 * q.um

        self.source = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                                    self.energies, np.array([0.2, 0.8]) * q.mm, self.ps,
                                    self.angle_step)

    @slow
    def test_bending_magnet_approx(self):
        source_2 = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                                 self.energies, np.array([0.2, 0.8]) * q.mm, self.ps,
                                 self.angle_step, profile_approx=False)

        for e in self.energies:
            profile = self.source.get_vertical_profile(e)
            profile_2 = source_2.get_vertical_profile(e)
            perc = profile / profile_2

            # Allow 0.1 % difference
            np.testing.assert_allclose(perc, 1, rtol=1e-3)

    def test_transfer(self):
        shape = self.angle_step
        # No trajectory
        self.source.transfer(shape, self.ps, self.energies[0])
        # Width may be larger
        self.source.transfer((self.angle_step, 2 * self.angle_step), self.ps, self.energies[0])

        # Wrong shape
        bad_shape = 2 * self.angle_step
        self.assertRaises(ValueError, self.source.transfer, bad_shape, self.ps, self.energies[0])

        # With trajectory
        n = 16
        x = z = np.zeros(n)
        y = np.linspace(0, 1, n) * q.mm
        tr = Trajectory(zip(x, y, z) * q.mm, velocity=1 * q.mm / q.s)
        source = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                               self.energies, np.array([0.2, 0.8]) * q.mm, self.ps,
                               self.angle_step, trajectory=tr)
        im_0 = source.transfer(shape, self.ps, self.energies[0]).get()
        im_1 = source.transfer(shape, self.ps, self.energies[0], t=tr.time / 2).get()

        # There must be a difference between two different times and given trajectory
        self.assertGreater(np.abs(im_1 - im_0).max(), 0)
