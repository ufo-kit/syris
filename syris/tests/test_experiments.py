import quantities as q
from syris.devices.cameras import Camera
from syris.devices.lenses import Lens
from syris.devices.detectors import Detector
from syris.experiments import Experiment
from syris.imageprocessing import Tiler
from unittest import TestCase


class TestExperiments(TestCase):

    def setUp(self):
        lens = Lens(1.0, 1.0, 3.0, (1 * q.um, 1 * q.um))
        camera = Camera(1 * q.um, 0.1, 10, 1.0, 12, None, shape=(64, 64))
        detector = Detector(lens, camera)
        tiler = Tiler(camera.shape, (1, 1), outlier=True, supersampling=2)
        self.experiment = Experiment(None, None, detector, tiler)

    def test_bad_init(self):
        self.assertRaises(ValueError, Experiment, None, None, None, None,
                          samples=[1, 2], spatial_incoherence=True)

    def test_add_sample(self):
        self.experiment.add_sample(1, 10 * q.mm)
        self.assertRaises(ValueError, self.experiment.add_sample, 2, 10 * q.mm)

    def test_pixel_size(self):
        pixel_size = self.experiment.detector.pixel_size / \
            self.experiment.tiler.supersampling
        self.assertAlmostEqual(self.experiment.super_pixel_size, pixel_size)