"""Public precision selection and mesh/camera coordinate-frame regressions."""

import logging
import unittest

import numpy as np
import quantities as q

import syris
from syris import config as cfg
from syris.bodies.mesh import Mesh
from syris.devices.cameras import Camera
from syris.geometry import Trajectory

from .cuda_projection_util import box, cuda_or_skip


class TestPrecisionSelection(unittest.TestCase):
    def test_invalid_and_conflicting_modes_fail_before_initialization(self):
        for kwargs in (
            {"precision": "half"},
            {"precision": "float32", "double_precision": True},
            {"precision": "float64", "mixed_precision": True},
            {"precision": "mixed", "double_precision": True},
            {"precision": "mixed", "compute_backend": "opencl"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                syris.init(**kwargs)


class PrecisionModeCases:
    @classmethod
    def setUpClass(cls):
        cuda_or_skip()
        cls.old = {name: getattr(cfg, name) for name in ("BACKEND", "PRECISION", "UNIT", "OPENCL")}
        syris.init(compute_backend="cuda", precision=cls.mode, loglevel=logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.old.items():
            setattr(cfg, name, value)

    def scene(self, shift, unit=q.um, normals=False, vertices=None):
        if vertices is None:
            vertices = box(z0=-0.5, z1=0.5)
        mesh = Mesh(
            vertices.reshape(-1, 3).T * q.um,
            Trajectory([(0, 0, 0)] * q.um),
            center=None,
            use_normals=normals,
        )
        mesh.translate(np.asarray(shift) * unit)
        camera = Camera(
            pixel_size=np.array([0.125, 0.0625]) * q.um,
            shape=(7, 9),
            source_detector_distance=-4 * q.um,
        )
        camera.translate(np.asarray(shift) * unit + np.array([0, 0, -2]) * q.um)
        return mesh, camera

    def test_output_dtype_and_selected_mode(self):
        self.assertEqual(cfg.PRECISION.mode, self.mode)
        mesh, camera = self.scene([0, 0, 0])
        image = mesh.build_accelerator().project(camera=camera).get()
        self.assertEqual(image.dtype, np.float64 if self.mode == "float64" else np.float32)
        np.testing.assert_allclose(image, 1, rtol=2e-6)

    def test_common_translation_preserves_parallel_cone_and_adaptive_rays(self):
        # This displacement would erase the complete box if cast to Float32
        # before subtracting it. Mixed input units exercise unit conversion.
        for normals in (False, True):
            mesh, camera = self.scene([0, 0, 0], normals=normals)
            moved, moved_camera = self.scene([1024, -2048, 4096], unit=q.mm, normals=normals)
            accel, moved_accel = mesh.build_accelerator(), moved.build_accelerator()
            for parallel in (True, False):
                for depth, sampling in ((0, "local"), (1, "local"), (1, "staged")):
                    with self.subTest(
                        normals=normals, parallel=parallel, depth=depth, sampling=sampling
                    ):
                        kwargs = dict(parallel=parallel, iterations=depth, sampling_mode=sampling)
                        expected = accel.project(camera=camera, **kwargs).get()
                        actual = moved_accel.project(camera=moved_camera, **kwargs).get()
                        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)

    def test_camera_motion_reuses_mesh_frame(self):
        mesh, camera = self.scene([17, -23, 11])
        accel = mesh.build_accelerator()
        np.testing.assert_allclose(accel.project(camera=camera).get(), 1)
        camera.translate(np.array([10, 0, 0]) * q.um)
        self.assertIs(mesh.build_accelerator(), accel)
        np.testing.assert_array_equal(accel.project(camera=camera).get(), 0)

    def test_rotated_camera_and_mesh_preserve_physical_rays(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(412).normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array([32, -17, 9]) * 1e-6  # Transform translations are metres.
        for normals in (False, True):
            mesh, camera = self.scene([0, 0, 0], normals=normals)
            moved, moved_camera = self.scene([0, 0, 0], normals=normals)
            moved.transform_matrix = transform @ moved.transform_matrix
            moved_camera.transform_matrix = transform @ moved_camera.transform_matrix
            for parallel in (False, True):
                with self.subTest(normals=normals, parallel=parallel):
                    expected = (
                        mesh.build_accelerator().project(camera=camera, parallel=parallel).get()
                    )
                    actual = (
                        moved.build_accelerator()
                        .project(camera=moved_camera, parallel=parallel)
                        .get()
                    )
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_cached_local_geometry_follows_successive_scan_poses(self):
        mesh, camera = self.scene([0, 0, 0], vertices=box(x0=-1, x1=1, z0=-0.5, z1=0.5))
        for angle, thickness in ((0, 1), (90, 2), (0, 1)):
            with self.subTest(angle=angle):
                mesh.clear_transformation()
                mesh.rotate(angle * q.deg, np.array([0, 1, 0]))
                image = mesh.build_accelerator().project(camera=camera).get()
                np.testing.assert_allclose(image, thickness, rtol=2e-6)


class TestPublicFloat32(PrecisionModeCases, unittest.TestCase):
    mode = "float32"


class TestPublicMixed(PrecisionModeCases, unittest.TestCase):
    mode = "mixed"


class TestPublicFloat64(PrecisionModeCases, unittest.TestCase):
    mode = "float64"
