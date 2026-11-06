"""Run directly as well as in discovery, including against an installed wheel."""

from importlib.resources import files
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import quantities as q

from syris import config as cfg
from syris.devices.cameras import make_pco_dimax
from syris.gpu.util import get_source
from syris.materials import Material


class TestPackageResources(TestCase):
    def test_cuda_sources_and_transitive_headers_are_available(self):
        directory = files("syris").joinpath("gpu", "cuda")
        for name in (
            "WatertightRay.cu",
            "Ray.cu",
            "source.cu",
            "Legacy.cu",
            "Commons.cuh",
            "WatertightRay.cuh",
            "Ray.cuh",
            "source.cuh",
            "PixelScheduler.cuh",
            "staged.cuh",
        ):
            with self.subTest(name=name):
                self.assertTrue(directory.joinpath(name).read_text(encoding="utf-8").strip())

    def test_opencl_source_and_template_load(self):
        source = get_source(["vcomplex.cl", "physics.cl"], precision_sensitive=False)
        self.assertIn("__kernel", source)
        self.assertTrue(get_source(["varconvolution.in"], precision_sensitive=False).strip())

    def test_camera_factory_reads_packaged_calibration(self):
        with patch.object(cfg, "UNIT", q.um), patch.object(cfg, "PRECISION", cfg.Precision(False)):
            camera = make_pco_dimax()
            np.testing.assert_array_equal(camera.shape, (2048, 2048))
            efficiency = camera.get_quantum_efficiency(500 * q.nm)
            self.assertTrue(np.isfinite(efficiency))
            self.assertGreater(efficiency, 0)

    def test_material_module_imports_without_distutils(self):
        self.assertTrue(callable(Material))


if __name__ == "__main__":
    main(verbosity=2)
