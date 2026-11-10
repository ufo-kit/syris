"""Real OBJ/VTKHDF frame I/O and mesh conversion; no GPU is needed."""

import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import quantities as q
import vtk
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

from syris import config as cfg
from syris.bodies.mesh import Mesh
from syris.bodies.meshreader import MeshAnimationReader, WavefrontAnimationReader
from syris.geometry import Trajectory


def tetrahedron(offset=0):
    points = np.array(
        [[0.123456789012345, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    points[:, 0] += offset
    faces = np.array([[3, 0, 2, 1], [3, 0, 1, 3], [3, 1, 2, 3], [3, 2, 0, 3]])
    return pv.PolyData(points, faces)


def triangle_vertices(mesh):
    return mesh.points[mesh.faces.reshape(-1, 4)[:, 1:]].reshape(-1, 3)


def write_obj(path, mesh):
    # Use enough digits to avoid introducing a second rounding during fixture export.
    with path.open("w") as stream:
        for point in mesh.points:
            stream.write("v " + " ".join(format(x, ".17g") for x in point) + "\n")
        for face in mesh.faces.reshape(-1, 4)[:, 1:]:
            stream.write("f " + " ".join(str(int(x) + 1) for x in face) + "\n")


class TemporalPolyData(VTKPythonAlgorithmBase):
    """Small independent VTK pipeline used by the native temporal writer."""

    def __init__(self, frames, times):
        super().__init__(nInputPorts=0, outputType="vtkPolyData")
        self.frames = frames
        self.times = tuple(times)

    def RequestInformation(self, request, in_info, out_info):
        info = out_info.GetInformationObject(0)
        pipeline = vtk.vtkStreamingDemandDrivenPipeline
        info.Set(pipeline.TIME_STEPS(), self.times, len(self.times))
        info.Set(pipeline.TIME_RANGE(), (self.times[0], self.times[-1]), 2)
        return 1

    def RequestData(self, request, in_info, out_info):
        info = out_info.GetInformationObject(0)
        time = info.Get(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP())
        output = vtk.vtkPolyData.GetData(out_info)
        output.ShallowCopy(self.frames[self.times.index(time)])
        output.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), time)
        return 1


def write_hdf(path, frames, times=None, compression=0):
    writer = vtk.vtkHDFWriter()
    writer.SetFileName(str(path))
    writer.SetCompressionLevel(compression)
    if times is None:
        writer.SetInputData(frames[0])
    else:
        source = TemporalPolyData(frames, times)
        writer.SetInputConnection(source.GetOutputPort())
        writer.SetWriteAllTimeSteps(True)
    if writer.Write() != 1:
        raise RuntimeError("Failed to write the VTKHDF test fixture")


class TestOBJAnimation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = pathlib.Path(directory.name)

    def test_legacy_entry_point_order_shapes_and_restart(self):
        expected = [tetrahedron(1), tetrahedron(2)]
        for index in (1, 0):
            write_obj(self.folder / f"{index:03d}.obj", expected[index])
        reader = WavefrontAnimationReader(folder=self.folder)
        self.assertEqual(len(reader), 2)
        self.assertIsNone(reader.time_values)
        self.assertIsNone(reader.frame_index)
        for name in ("vertices", "normals", "bounds"):
            with self.assertRaises(AttributeError):
                getattr(reader, name)
        for _ in range(2):
            for index, frame in enumerate(reader.iter_frames()):
                self.assertIs(frame, reader)
                self.assertEqual(frame.frame_index, index)
                self.assertIsNone(frame.frame_time)
                self.assertEqual(frame.vertices.dtype, np.float32)
                self.assertEqual(frame.normals.shape, (4, 3))
                np.testing.assert_array_equal(
                    frame.vertices, triangle_vertices(expected[index]).astype(np.float32)
                )
                np.testing.assert_array_equal(
                    frame.bounds, np.array(expected[index].bounds, np.float32)
                )

        # The benchmark supplies its own ordering through the existing public list.
        reader.filenames = reader.filenames[::-1][:1]
        self.assertEqual(len(reader), 1)
        np.testing.assert_array_equal(
            next(iter(reader)).vertices, triangle_vertices(expected[1]).astype(np.float32)
        )

    def test_input_errors_and_failed_read_do_not_advance_frame(self):
        with self.assertRaises(FileNotFoundError):
            MeshAnimationReader(self.folder)
        path = self.folder / "frame.obj"
        write_obj(path, tetrahedron())
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            MeshAnimationReader(self.folder, dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "OBJ folder"):
            MeshAnimationReader(path)
        reader = MeshAnimationReader(self.folder)
        for index in (-1, 1):
            with self.assertRaises(IndexError):
                reader.read_frame(index)
        with self.assertRaises(TypeError):
            reader.read_frame(0.5)
        reader.read_frame(0)
        saved = reader.vertices.copy()
        path.unlink()
        with self.assertRaises(OSError):
            reader.read_frame(0)
        np.testing.assert_array_equal(reader.vertices, saved)


@unittest.skipUnless(hasattr(vtk, "vtkHDFWriter"), "VTK with vtkHDFWriter is needed for fixtures")
class TestVTKHDFAnimation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = pathlib.Path(directory.name)
        self.path = self.folder / "animation.vtkhdf"
        self.frames = [tetrahedron(), tetrahedron(0.25), tetrahedron(2)]
        self.frames[1].points[:, 2] *= 1.125
        # The last frame has a different topology and number of triangles.
        self.frames[2] = self.frames[2].merge(tetrahedron(4), merge_points=False)
        self.times = (0.0, 0.125, 2.5)

    def test_temporal_geometry_precision_and_nonsequential_access(self):
        for compression in (0, 1):
            write_hdf(self.path, self.frames, self.times, compression=compression)
            for dtype in (np.float32, np.float64):
                with self.subTest(compression=compression, dtype=dtype):
                    reader = MeshAnimationReader(self.path, dtype=dtype)
                    self.assertEqual(len(reader), 3)
                    self.assertEqual(reader.time_values, self.times)
                    with self.assertRaises(AttributeError):
                        _ = reader.vertices
                    for index in (0, 2, 1, 0):
                        frame = reader.read_frame(index)
                        self.assertIs(frame, reader)
                        self.assertEqual(frame.frame_index, index)
                        self.assertEqual(frame.frame_time, self.times[index])
                        self.assertEqual(frame.vertices.dtype, dtype)
                        self.assertEqual(frame.normals.dtype, dtype)
                        self.assertEqual(frame.bounds.dtype, dtype)
                        np.testing.assert_array_equal(
                            frame.vertices, triangle_vertices(self.frames[index]).astype(dtype)
                        )
                        np.testing.assert_array_equal(
                            frame.bounds, np.array(self.frames[index].bounds, dtype)
                        )
                    for _ in range(2):
                        self.assertEqual([f.frame_time for f in reader], list(self.times))
                    # Release the archive before overwriting it with another compression setting.
                    del frame, reader

    def test_obj_and_hdf_frames_match_and_previous_arrays_are_owned(self):
        write_hdf(self.path, self.frames, self.times)
        for index, frame in enumerate(self.frames):
            write_obj(self.folder / f"{index:03d}.obj", frame)
        obj = WavefrontAnimationReader(self.folder)
        hdf = WavefrontAnimationReader(self.path)
        retained = []
        for a, b in zip(obj, hdf):
            np.testing.assert_array_equal(a.vertices, b.vertices)
            np.testing.assert_array_equal(a.bounds, b.bounds)
            np.testing.assert_allclose(a.normals, b.normals, atol=1e-7, rtol=0)
            retained.append((b.vertices, b.normals, b.bounds))
        for index, (vertices, normals, bounds) in enumerate(retained):
            np.testing.assert_array_equal(
                vertices, triangle_vertices(self.frames[index]).astype(np.float32)
            )
            np.testing.assert_array_equal(bounds, np.array(self.frames[index].bounds, np.float32))
            self.assertEqual(normals.shape, (self.frames[index].n_cells, 3))
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)

    def test_static_hdf_and_stored_normals(self):
        mesh = tetrahedron()
        triangles = triangle_vertices(mesh).reshape(-1, 3, 3)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        mesh.cell_data["Normals"] = normals
        mesh.cell_data.active_normals_name = "Normals"
        path = self.folder / "static.hdf"
        write_hdf(path, [mesh])
        reader = MeshAnimationReader(path, dtype=np.float64)
        self.assertEqual(len(reader), 1)
        self.assertIsNone(reader.time_values)
        frame = next(iter(reader))
        self.assertIsNone(frame.frame_time)
        np.testing.assert_array_equal(frame.vertices, triangle_vertices(mesh))
        np.testing.assert_array_equal(frame.normals, normals)

    def test_polygon_triangulation_and_mesh_construction(self):
        mesh = pv.Cube()
        write_hdf(self.path, [mesh])
        reader = MeshAnimationReader(self.path, dtype=np.float64)
        frame = reader.read_frame(0)
        self.assertEqual(frame.vertices.shape, (36, 3))
        self.assertEqual(frame.normals.shape, (12, 3))
        with patch.multiple(cfg, PRECISION=cfg.Precision(True), UNIT=q.um):
            body = Mesh(
                frame.vertices.T * q.um,
                Trajectory([(0, 0, 0)] * q.um),
                center=None,
                bounds=frame.bounds.copy() * q.um,
            )
            self.assertEqual(body.num_triangles, 12)
            np.testing.assert_array_equal(body.extrema.magnitude.ravel(), frame.bounds)

    def test_invalid_archives_and_unsupported_data(self):
        with self.assertRaises(FileNotFoundError):
            MeshAnimationReader(self.path)
        self.path.write_text("not an HDF file")
        with self.assertRaisesRegex(ValueError, "readable VTKHDF"):
            MeshAnimationReader(self.path)
        write_hdf(self.path, [tetrahedron().cast_to_unstructured_grid()])
        with self.assertRaisesRegex(ValueError, "vtkPolyData"):
            MeshAnimationReader(self.path)


if __name__ == "__main__":
    unittest.main()
