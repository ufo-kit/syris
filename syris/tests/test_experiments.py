import numpy as np
import quantities as q
import syris
from syris.bodies.isosurfaces import MetaBall
from syris.devices.cameras import Camera
from syris.devices.lenses import Lens
from syris.devices.detectors import Detector
from syris.geometry import Trajectory
from syris.tests import SyrisTest
from syris.experiments import Experiment


class TestExperiments(SyrisTest):

    def setUp(self):
        syris.init()
        lens = Lens(3., f_number=1.4, focal_length=100.0 * q.mm)
        camera = Camera(1 * q.um, 0.1, 10, 1.0, 12, (64, 64))
        detector = Detector(None, lens, camera)
        ps = detector.pixel_size
        t = np.linspace(0, 1, 10) * q.mm
        x = t
        y = np.zeros(len(t))
        z = np.zeros(len(t))
        points = zip(x, y, z) * q.mm
        mb_0 = MetaBall(Trajectory(points, pixel_size=ps, furthest_point=1 * q.um,
                                   velocity=1 * q.mm / q.s), 1 * q.um)
        mb_1 = MetaBall(Trajectory(points, pixel_size=ps, furthest_point=1 * q.um,
                                   velocity=2 * q.mm / q.s), 1 * q.um)
        self.experiment = Experiment([mb_0, mb_1], None, detector, 0 * q.m, None)

    def test_pixel_size(self):
        self.assertAlmostEqual(1, self.experiment.time.simplified.magnitude)
