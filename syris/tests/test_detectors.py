# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

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
