"""Analytic material regions and precision checks for optional mesh repair.

No GPU is needed. Install .[mesh-repair] to run the libigl integration cases.
"""

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import quantities as q

from syris import config as cfg
from syris.bodies.mesh import Mesh
from syris.bodies.meshreader import PyvistaReader
from syris.bodies.meshrepair import _round_float32, repair_mesh
from syris.geometry import Trajectory

try:
    import igl
    from igl.copyleft import cgal
except ImportError:
    cgal = None


def cube(half=1.0, center=(0, 0, 0), inward=False):
    vertices = (
        half
        * np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        + center
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces[:, ::-1].copy() if inward else faces


def combine(*parts):
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.vstack(vertices), np.vstack(faces)


def polydata(vertices, faces):
    return pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)))


def volume(vertices, faces):
    t = vertices[faces]
    return np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6


def close_sloping_boxes():
    # Disjoint solids with a narrow, tilted gap. Nearest Float32 rounding
    # makes their opposing faces cross, without merging any vertices.
    vertices, faces = combine(cube(), cube(center=(0.5, 0.5, 2 + 2**-27)))
    vertices[:, 2] += 0.13 * vertices[:, 0] + 0.07 * vertices[:, 1]
    return vertices, faces


class TestOptionalMeshRepair(unittest.TestCase):
    def test_rounding_requires_float32_repair(self):
        surface = polydata(*cube())
        for kwargs in ({}, {"repair": True, "dtype": np.float64}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, "float32_rounding requires"),
            ):
                PyvistaReader(surface, float32_rounding=True, **kwargs)
        with self.assertRaisesRegex(ValueError, "float32_rounding requires"):
            repair_mesh(*cube(), float32_rounding=True)

    def test_loader_works_without_libigl_and_repair_reports_install_command(self):
        surface = polydata(*cube())
        with patch.dict(sys.modules, {"igl": None, "igl.copyleft": None}):
            reader = PyvistaReader(surface)
            self.assertEqual(reader.vertices.shape, (3, 36))
            with self.assertRaisesRegex(ImportError, r"uv pip install.*mesh-repair"):
                PyvistaReader(surface, repair=True)

    def test_loader_honors_configured_precision(self):
        vertices, faces = cube(center=(1e8, 0, 0))
        with patch.multiple(cfg, PRECISION=cfg.Precision(True), UNIT=q.um):
            mesh = Mesh.from_file(
                polydata(vertices, faces), Trajectory([(0, 0, 0)] * q.um), unit=q.um
            )
            self.assertEqual(mesh._current.dtype, np.float64)
            self.assertEqual(np.ptp(mesh._current[0]), 2)


@unittest.skipIf(cgal is None, "Install .[mesh-repair] for libigl integration tests")
class TestMeshRepair(unittest.TestCase):
    def assert_region(self, vertices, faces, expected_volume, queries, winding):
        self.assertAlmostEqual(volume(vertices, faces), expected_volume, places=12)
        actual = igl.winding_number(
            vertices.astype(np.float64), faces, np.asarray(queries, dtype=np.float64)
        )
        np.testing.assert_allclose(actual, winding, atol=1e-12, rtol=0)

    def test_hollow_solid_with_nested_island_and_small_disconnected_component(self):
        vertices, faces = combine(
            cube(), cube(0.5, inward=True), cube(0.25), cube(0.125, center=(3, 0, 0))
        )
        for dtype, rounding in ((np.float32, False), (np.float64, False), (np.float32, True)):
            with self.subTest(dtype=dtype, rounding=rounding):
                v, f = repair_mesh(vertices, faces, dtype=dtype, float32_rounding=rounding)
                self.assertEqual(v.dtype, dtype)
                self.assert_region(
                    v,
                    f,
                    7.125 + 0.25**3,
                    [[0, 0, 0], [0.375, 0, 0], [0.75, 0, 0], [2, 0, 0], [3, 0, 0]],
                    [1, 0, 1, 0, 1],
                )

    def test_overlapping_solids_become_one_material_region(self):
        vertices, faces = combine(cube(), cube(center=(1, 0, 0)))
        v, f = repair_mesh(vertices, faces)
        self.assert_region(
            v, f, 12, [[-0.5, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [3, 0, 0]], [1, 1, 1, 0]
        )

    def test_negative_winding_is_excluded_instead_of_taking_absolute_value(self):
        vertices, faces = combine(cube(), cube(center=(1, 0, 0), inward=True))
        v, f = repair_mesh(vertices, faces)
        self.assert_region(v, f, 4, [[-0.5, 0, 0], [0.5, 0, 0], [1.5, 0, 0]], [1, 0, 0])

    def test_exact_welding_and_degenerate_removal_without_mutating_input(self):
        vertices, faces = cube()
        vertices = vertices[faces].reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        faces = np.vstack((faces, [0, 0, 1]))
        original_vertices, original_faces = vertices.copy(), faces.copy()
        v, f = repair_mesh(vertices, faces)
        self.assert_region(v, f, 8, [[0, 0, 0], [2, 0, 0]], [1, 0])
        np.testing.assert_array_equal(vertices, original_vertices)
        np.testing.assert_array_equal(faces, original_faces)

    def test_open_boundaries_and_flipped_faces_are_rejected(self):
        vertices, faces = cube()
        flipped = faces.copy()
        flipped[0] = flipped[0, ::-1]
        for invalid in (faces[:-1], flipped):
            with (
                self.subTest(faces=invalid),
                self.assertRaisesRegex(ValueError, "closed, consistently oriented"),
            ):
                repair_mesh(vertices, invalid)

    def test_all_inward_input_does_not_silently_turn_into_an_exterior(self):
        with self.assertRaisesRegex(ValueError, "no positive-winding material"):
            repair_mesh(*cube(inward=True))

    def test_float32_collapse_requires_double_precision(self):
        vertices, faces = cube(center=(1e8, 0, 0))
        for rounding in (False, True):
            with (
                self.subTest(rounding=rounding),
                self.assertRaisesRegex(ValueError, r"float32.*double_precision=True"),
            ):
                repair_mesh(vertices, faces, dtype=np.float32, float32_rounding=rounding)
        v, _ = repair_mesh(vertices, faces)
        self.assertEqual(np.ptp(v[:, 0]), 2)

    def test_float32_contact_between_disconnected_shells_is_rejected(self):
        # The gap is representable in Float64, but disappears on casting.
        vertices, faces = combine(cube(), cube(center=(2 + 2**-25, 0, 0)))
        for rounding in (False, True):
            with (
                self.subTest(rounding=rounding),
                self.assertRaisesRegex(ValueError, r"float32.*double_precision=True"),
            ):
                repair_mesh(vertices, faces, dtype=np.float32, float32_rounding=rounding)
        v, f = repair_mesh(vertices, faces)
        self.assertAlmostEqual(volume(v, f), 16, places=12)
        # Direct solid-angle summation loses some digits this close to a face.
        winding = igl.winding_number(v, f, np.array([[1 + 2**-26, 0, 0]]))
        np.testing.assert_allclose(winding, 0, rtol=0, atol=1e-8)

    def test_bounded_rounding_resolves_crossings_and_preserves_connectivity(self):
        raw, faces = close_sloping_boxes()
        saved_raw, saved_faces = raw.copy(), faces.copy()
        for scale in (2**-30, 1, 2**30):
            with self.subTest(scale=scale):
                original, reference_faces = repair_mesh(raw * scale, faces)
                with self.assertRaisesRegex(ValueError, "self-intersections"):
                    repair_mesh(raw * scale, faces, dtype=np.float32)
                v, f = repair_mesh(raw * scale, faces, dtype=np.float32, float32_rounding=True)
                again, _ = repair_mesh(raw * scale, faces, dtype=np.float32, float32_rounding=True)
                np.testing.assert_array_equal(v, again)
                np.testing.assert_array_equal(f, reference_faces)
                self.assertEqual(v.dtype, np.float32)
                self.assertEqual(len(np.unique(v, axis=0)), len(original))
                self.assertFalse(
                    len(
                        cgal.remesh_self_intersections(v.astype(np.float64), f, detect_only=True)[2]
                    )
                )
                # Every changed coordinate must be an immediate neighbor of
                # the exact value, not an unbounded displacement or smoothing.
                below = v.astype(np.float64) < original
                above = v.astype(np.float64) > original
                self.assertTrue(
                    np.all(np.nextafter(v, np.float32(np.inf))[below] >= original[below])
                )
                self.assertTrue(
                    np.all(np.nextafter(v, np.float32(-np.inf))[above] <= original[above])
                )
                exact = original == original.astype(np.float32).astype(np.float64)
                np.testing.assert_array_equal(v[exact], original[exact])
                before, after = original[f], v[f].astype(np.float64)
                n0 = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
                n1 = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
                self.assertTrue(np.all(np.einsum("ij,ij->i", n0, n1) > 0))
                np.testing.assert_allclose(volume(v.astype(np.float64) / scale, f), 16, rtol=2e-7)
        np.testing.assert_array_equal(raw, saved_raw)
        np.testing.assert_array_equal(faces, saved_faces)

    def test_rounding_search_limits_raise_instead_of_returning_invalid_mesh(self):
        v, f = repair_mesh(*close_sloping_boxes())
        for limits in ({"max_vertices": 0}, {"max_patch_faces": 0}, {"max_evaluations": 0}):
            with (
                self.subTest(limits=limits),
                self.assertRaisesRegex(ValueError, "search limit|no valid configuration"),
            ):
                _round_float32(cgal, v, f, **limits)

    def test_mesh_loader_uses_rounded_coordinates_and_recomputes_normals(self):
        surface = polydata(*close_sloping_boxes())
        surface.cell_data["Normals"] = np.ones((surface.n_cells, 3))
        surface.cell_data.active_normals_name = "Normals"
        points, faces = repair_mesh(*close_sloping_boxes(), dtype=np.float32, float32_rounding=True)
        with patch.multiple(cfg, PRECISION=cfg.Precision(False), UNIT=q.um):
            mesh = Mesh.from_file(
                surface,
                Trajectory([(0, 0, 0)] * q.um),
                unit=q.um,
                use_normals=True,
                repair=True,
                float32_rounding=True,
            )
            triangles = mesh._current[:3].T.reshape(-1, 3, 3)
            np.testing.assert_array_equal(triangles, points[faces])
            expected = points[faces].astype(np.float64)
            normals = np.cross(expected[:, 1] - expected[:, 0], expected[:, 2] - expected[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            np.testing.assert_allclose(np.asarray(mesh.normals), normals, atol=1e-7)
        np.testing.assert_array_equal(surface.points, close_sloping_boxes()[0])
        np.testing.assert_array_equal(surface.cell_data["Normals"], 1)

    def test_loader_replaces_stale_normals_and_preserves_cavity_orientation(self):
        surface = polydata(*combine(cube(), cube(0.5, inward=True)))
        surface.cell_data["Normals"] = np.ones((surface.n_cells, 3))
        surface.cell_data.active_normals_name = "Normals"
        original_points, original_faces = surface.points.copy(), surface.faces.copy()
        with patch.multiple(cfg, PRECISION=cfg.Precision(True), UNIT=q.um):
            mesh = Mesh.from_file(
                surface, Trajectory([(0, 0, 0)] * q.um), unit=q.um, use_normals=True, repair=True
            )
            vertices = mesh._current[:3].T.reshape(-1, 3, 3)
            expected = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            expected /= np.linalg.norm(expected, axis=1)[:, None]
            np.testing.assert_allclose(np.asarray(mesh.normals), expected, atol=1e-15)
            centers = vertices.mean(axis=1)
            signs = np.einsum("ij,ij->i", centers, np.asarray(mesh.normals))
            cavity = np.max(np.abs(vertices), axis=(1, 2)) < 1
            self.assertTrue(np.all(signs[cavity] < 0))
            self.assertTrue(np.all(signs[~cavity] > 0))
        np.testing.assert_array_equal(surface.points, original_points)
        np.testing.assert_array_equal(surface.faces, original_faces)
        np.testing.assert_array_equal(surface.cell_data["Normals"], 1)


if __name__ == "__main__":
    unittest.main()
