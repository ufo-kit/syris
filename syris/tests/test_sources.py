import numpy as np
import quantities as q
import syris
from syris.devices.sources import BendingMagnet
from syris.geometry import Trajectory
from syris.tests import SyrisTest, slow


class TestSources(SyrisTest):

    def setUp(self):
        syris.init()
        self.dE = 0.1 * q.keV
        self.energies = np.arange(14.8, 15, self.dE.magnitude) * q.keV
        self.trajectory = Trajectory([(0, 0, 0)] * q.m)
        self.ps = 10 * q.um

        self.source = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                                    self.dE, np.array([0.2, 0.8]) * q.mm, self.ps,
                                    self.trajectory)

    @slow
    def test_bending_magnet_approx(self):
        source_2 = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                                 self.dE, np.array([0.2, 0.8]) * q.mm, self.ps,
                                 self.trajectory, profile_approx=False)

        for e in self.energies:
            u_0 = self.source.transfer((16, 16), self.ps, e, t=0 * q.s).get().real
            u_1 = source_2.transfer((16, 16), self.ps, e, t=0 * q.s).get().real
            perc = u_0 / u_1

            # Allow 0.1 % difference
            np.testing.assert_allclose(perc, 1, rtol=1e-3)

    def test_transfer(self):
        shape = 10
        # No trajectory
        self.source.transfer(shape, self.ps, self.energies[0], t=0 * q.s)
        # Width may be larger
        self.source.transfer((shape, 2 * shape), self.ps, self.energies[0], t=0 * q.s)

        # With trajectory
        n = 16
        x = z = np.zeros(n)
        y = np.linspace(0, 1, n) * q.mm
        tr = Trajectory(zip(x, y, z) * q.mm, pixel_size=10 * q.um, furthest_point=0*q.m,
                        velocity=1 * q.mm / q.s)
        source = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                               self.dE, np.array([0.2, 0.8]) * q.mm, self.ps, trajectory=tr)
        im_0 = source.transfer(shape, self.ps, self.energies[0], t=0 * q.s).get()
        im_1 = source.transfer(shape, self.ps, self.energies[0], t=tr.time / 2).get()

        # There must be a difference between two different times and given trajectory
        self.assertGreater(np.abs(im_1 - im_0).max(), 0)
