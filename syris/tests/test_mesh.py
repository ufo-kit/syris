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

import itertools
import numpy as np
import quantities as q
from syris.geometry import Trajectory, X_AX, Y_AX
from syris.bodies.mesh import Mesh, make_cube
from syris.tests import default_syris_init, SyrisTest
from syris.util import get_magnitude


class TestMesh(SyrisTest):
    def setUp(self):
        default_syris_init()
        self.triangles = make_cube()
        self.trajectory = Trajectory([(0, 0, 0)] * q.m)
        self.mesh = Mesh(self.triangles, self.trajectory)

    def shift_mesh(self, point):
        triangles = self.triangles + np.array(point)[:, np.newaxis] * point.units
        return Mesh(triangles, self.trajectory, center=None)

    def test_furthest_point(self):
        self.assertAlmostEqual(get_magnitude(self.mesh.furthest_point), np.sqrt(3))

    def test_bounding_box(self):
        bbox_points = get_magnitude(self.mesh.bounding_box.points).astype(int).tolist()
        seed = (-1, 1)
        for point in itertools.product(seed, seed, seed):
            self.assertTrue(list(point) in bbox_points)

    def test_num_triangles(self):
        self.assertEqual(12, self.mesh.num_triangles)

    def test_extrema(self):
        for endpoints in self.mesh.extrema:
            self.assertAlmostEqual(-1, get_magnitude(endpoints[0]))
            self.assertAlmostEqual(1, get_magnitude(endpoints[1]))

    def test_center_of_gravity(self):
        gt = (1, 2, 3) * q.m
        mesh = self.shift_mesh(gt)
        center = get_magnitude(mesh.center_of_gravity)
        np.testing.assert_almost_equal(get_magnitude(gt), center)

    def test_center_of_bbox(self):
        gt = (1, 2, 3) * q.m
        mesh = self.shift_mesh(gt)
        center = get_magnitude(mesh.center_of_bbox)
        np.testing.assert_almost_equal(get_magnitude(gt), center)

    def test_diff(self):
        gt = np.ones((3, 2)) * 2
        np.testing.assert_almost_equal(gt, get_magnitude(self.mesh.diff))

    def test_vectors(self):
        ba, ca = self.mesh.vectors
        a = self.triangles[:, ::3]
        b = self.triangles[:, 1::3]
        c = self.triangles[:, 2::3]

        ba_gt = (b - a).transpose()
        ca_gt = (c - a).transpose()

        np.testing.assert_almost_equal(ba_gt, ba)
        np.testing.assert_almost_equal(ca_gt, ca)

    def test_areas(self):
        np.testing.assert_almost_equal(2 * q.m ** 2, self.mesh.areas)

    def test_normals(self):
        v_0, v_1 = self.mesh.vectors
        gt = np.cross(v_0, v_1) * q.um

        np.testing.assert_almost_equal(gt, self.mesh.normals)

    def test_max_triangle_x_diff(self):
        self.assertEqual(2 * q.m, self.mesh.max_triangle_x_diff)

    def test_sort(self):
        self.mesh.sort()
        x = get_magnitude(self.mesh.triangles[0, 2::3])
        # Triangles must be sorted by the last vertex
        np.testing.assert_almost_equal(x, sorted(x))

        # The greatest is the last vertex in a triangle, all other vertices must be smaller
        x = get_magnitude(self.mesh.triangles[0, :])
        for i in range(0, len(x), 3):
            self.assertTrue(x[i] <= x[i + 2])
            self.assertTrue(x[i + 1] <= x[i + 2])

    def test_get_degenerate_triangles(self):
        triangles = get_magnitude(self.mesh.get_degenerate_triangles())

        for i in range(0, triangles.shape[1], 3):
            vertices = triangles[:, i : i + 3]
            x = vertices[0, :]
            y = vertices[1, :]
            x_any = np.any(x - x[0])
            y_any = np.any(y - y[0])
            self.assertTrue(not (x_any and y_any))

        self.mesh.rotate(45 * q.deg, X_AX)
        self.mesh.rotate(45 * q.deg, Y_AX)
        self.mesh.transform()

        self.assertEqual(0, self.mesh.get_degenerate_triangles().shape[1])
