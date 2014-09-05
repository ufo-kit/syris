import numpy as np
import quantities as q
from syris.devices.sources import BendingMagnet
from syris.tests import SyrisTest, slow


class TestSources(SyrisTest):

    @slow
    def test_bending_magnet_approx(self):
        angle_step = 10
        ps = 10 * q.um
        energies = np.arange(14.8, 15, 0.1) * q.keV

        source = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                               energies, np.array([0.2, 0.8]) * q.mm, ps,
                               angle_step)
        source_2 = BendingMagnet(2.5 * q.GeV, 150 * q.mA, 1.5 * q.T, 30 * q.m,
                                 energies, np.array([0.2, 0.8]) * q.mm, ps,
                                 angle_step, profile_approx=False)

        for e_i in range(len(energies)):
            profile = source.get_vertical_profile(e_i)
            profile_2 = source_2.get_vertical_profile(e_i)

            np.testing.assert_array_almost_equal(profile, profile_2, 0)
