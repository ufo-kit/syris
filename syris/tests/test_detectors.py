import quantities as q
from syris.devices.cameras import Camera
from syris.devices.detectors import Detector
from syris.devices.lenses import Lens
from syris.tests import SyrisTest


class TestDetector(SyrisTest):
    def setUp(self):
        self.lens = Lens(3.0, f_number=1.4, focal_length=100.0 * q.mm)
        self.camera = Camera(10 * q.um, 0.1, 10, 1, 12, None)
        self.detector = Detector(None, self.lens, self.camera)

    def test_pixel_size(self):
        self.assertEqual(self.detector.pixel_size, self.camera.pixel_size / self.lens.magnification)
