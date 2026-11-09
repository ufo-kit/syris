"""Exercise paired example entry points and their saved output conventions."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv
import quantities as q

from syris.tests.cuda_projection_util import cuda_or_skip

try:
    import imageio.v2 as imageio

    from examples import mesh, mesh_cuda, mesh_scan, mesh_scan_cuda
    from examples.compare_mesh_outputs import compare_directories
except ModuleNotFoundError as exc:
    if exc.name in ("imageio", "tqdm"):
        raise unittest.SkipTest("Mesh example tests require the examples extra") from exc
    raise

ROOT = Path(__file__).resolve().parents[2]


class TestMeshExampleOptions(unittest.TestCase):
    def test_named_precision_modes_resolve_existing_options(self):
        for module in (mesh_cuda, mesh_scan_cuda):
            for mode in ("float32", "mixed", "float64"):
                with self.subTest(module=module.__name__, mode=mode):
                    args = module.parse_args(["--precision", mode])
                    self.assertEqual(args.double_precision, mode == "float64")
                    self.assertEqual(args.mixed_precision, mode == "mixed")

    def test_named_modes_retain_backend_and_repair_constraints(self):
        for module, arguments in (
            (mesh, ["--precision", "mixed"]),
            (mesh_cuda, ["--precision", "float64", "--repair-mesh", "--float32-rounding"]),
            (mesh_cuda, ["--precision", "float32", "--double-precision"]),
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(arguments)

    def test_pairs_have_identical_defaults_except_backend(self):
        for left, right in ((mesh, mesh_cuda), (mesh_scan, mesh_scan_cuda)):
            a, b = vars(left.parse_args([])), vars(right.parse_args([]))
            self.assertEqual(a.pop("backend"), "opencl")
            self.assertEqual(b.pop("backend"), "cuda")
            self.assertEqual(a, b)

    def test_scan_angles_exclude_endpoint_and_allow_one_projection(self):
        args = mesh_scan.parse_args(["--num-projections", "4"])
        np.testing.assert_array_equal(
            mesh_scan.projection_angles(args, 32).rescale(q.deg), [0, 45, 90, 135]
        )
        args.num_projections = 1
        np.testing.assert_array_equal(mesh_scan.projection_angles(args, 32).rescale(q.deg), [0])

    def test_unsupported_sampling_modes_fail_before_initializing(self):
        cases = [
            (mesh, ["--subsampling", "1"]),
            (mesh_scan, ["--conebeam"]),
            (mesh_cuda, ["--supersampling-projection", "2"]),
            (mesh_scan_cuda, ["--double-precision", "--subsampling", "3"]),
        ]
        for module, arguments in cases:
            with self.subTest(module=module.__name__), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    module.parse_args(arguments)
                self.assertEqual(caught.exception.code, 2)

    def test_comparison_rejects_nonfinite_mismatched_and_differently_scaled_data(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            left, right = Path(tmp) / "left", Path(tmp) / "right"
            left.mkdir()
            right.mkdir()
            a = np.ones((3, 5), dtype=np.float32) * 2e-6
            imageio.imwrite(left / "projection.tif", a)
            imageio.imwrite(right / "projection.tif", a)
            self.assertTrue(compare_directories(left, right))
            for b in (a * 1e6, a.astype(np.float64), a[:, :3], a * np.nan):
                imageio.imwrite(right / "projection.tif", b)
                self.assertFalse(compare_directories(left, right))
            imageio.imwrite(right / "extra.tif", a)
            self.assertFalse(compare_directories(left, right))


class TestMeshExampleBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cuda_or_skip()
        import pyopencl as cl

        if not any(p.get_devices() for p in cl.get_platforms()):
            raise unittest.SkipTest("OpenCL unavailable")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        # Off-center, asymmetric closed mesh: exposes flips, rotation signs and centering errors.
        self.source = self.path / "asymmetric.ply"
        surface = pv.ParametricEllipsoid(0.93, 0.57, 0.38, u_res=19, v_res=13)
        surface.translate([0.31, -0.27, 0.19], inplace=True)
        surface.save(self.source)

    def run_example(self, module, arguments):
        result = subprocess.run(
            [sys.executable, "-m", "examples." + module, *map(str, arguments)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def assert_outputs_match(self, left, right, dtype):
        names = sorted(p.relative_to(left) for p in left.rglob("*.tif"))
        self.assertTrue(names)
        self.assertEqual(names, sorted(p.relative_to(right) for p in right.rglob("*.tif")))
        for name in names:
            a, b = imageio.imread(left / name), imageio.imread(right / name)
            self.assertEqual(a.dtype, dtype)
            self.assertEqual(b.dtype, dtype)
            self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all(), str(name))
            self.assertGreater(a.max(), 0, str(name))
            # Five picometres is <1e-5 of this fixture's peak thickness; small
            # chords near the silhouette amplify Float32 intersection roundoff.
            atol, rtol = (5e-12, 2e-5) if dtype == np.float32 else (2e-19, 2e-11)
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=str(name))
        return names

    def test_rectangular_rotating_mesh_and_uniform_sampling(self):
        for double, ss in ((False, 1), (True, 2)):
            with self.subTest(double=double, ss=ss):
                outputs = []
                for suffix in ("", "_cuda"):
                    directory = self.path / (f"mesh_{double}" + suffix)
                    directory.mkdir()
                    options = [
                        "--input",
                        self.source,
                        "--n-y",
                        37,
                        "--n-x",
                        43,
                        "--pixel-size",
                        90,
                        "--pixel-size-units",
                        "nm",
                        "--x-rotate",
                        17,
                        "--y-rotate",
                        29,
                        "--num-y-rotations",
                        3,
                        "--supersampling",
                        ss,
                        "--no-show",
                        "--projection-filename",
                        directory / "projection",
                    ]
                    if double:
                        options.append("--double-precision")
                    self.run_example("mesh" + suffix, options)
                    outputs.append(directory)
                names = self.assert_outputs_match(*outputs, np.float64 if double else np.float32)
                self.assertEqual(len(names), 3)
                self.assertEqual(imageio.imread(outputs[0] / names[0]).shape, (37, 43))

    def test_cube_thickness_units_and_subpixel_average(self):
        # Deliberately avoid shared-edge rays: those expose legacy OpenCL predicate failures.
        for suffix in ("", "_cuda"):
            prefix = self.path / ("cube" + suffix)
            self.run_example(
                "mesh" + suffix,
                [
                    "--n-y",
                    19,
                    "--n-x",
                    27,
                    "--pixel-size",
                    0.13,
                    "--translate",
                    1.713,
                    1.027,
                    "--supersampling",
                    3,
                    "--no-show",
                    "--projection-filename",
                    prefix,
                ],
            )
            actual = imageio.imread(str(prefix) + "_00000.tif")
            x = (np.arange(27 * 3) + 0.5) * 0.13 / 3 - 1.713
            y = (np.arange(19 * 3) + 0.5) * 0.13 / 3 - 1.027
            expected = ((np.abs(y[:, None]) < 1) & (np.abs(x[None, :]) < 1)) * 2e-6
            expected = expected.reshape(19, 3, 27, 3).mean(axis=(1, 3))
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-6)

    def test_scan_axes_angles_and_single_projection(self):
        for double, count in ((False, 1), (True, 4)):
            with self.subTest(double=double, count=count):
                outputs = []
                for suffix in ("", "_cuda"):
                    directory = self.path / (f"scan_{double}" + suffix)
                    options = [
                        "--input",
                        self.source,
                        "--n",
                        29,
                        "--num-projections",
                        count,
                        "--rotation-angle",
                        137,
                        "--lamino-angle",
                        23,
                        "--rotation-axis",
                        "y",
                        "z",
                        "--pixel-size",
                        90,
                        "--supersampling",
                        2,
                        "--out-directory",
                        directory,
                    ]
                    if double:
                        options.append("--double-precision")
                    self.run_example("mesh_scan" + suffix, options)
                    outputs.append(directory)
                names = self.assert_outputs_match(*outputs, np.float64 if double else np.float32)
                self.assertEqual(len(names), 2 * count)

    def test_ground_truth_uses_same_slicer_and_handles_last_chunk(self):
        outputs = []
        for suffix in ("", "_cuda"):
            directory = self.path / ("truth" + suffix)
            result = self.run_example(
                "mesh_scan" + suffix,
                [
                    "--n",
                    7,
                    "--pixel-size",
                    400,
                    "--supersampling",
                    2,
                    "--z-chunk",
                    6,
                    "--make-gt",
                    "--out-directory",
                    directory,
                ],
            )
            self.assertIn("Ground-truth slicing uses OpenCL", result.stderr)
            outputs.append(directory)
        names = sorted(p.relative_to(outputs[0]) for p in outputs[0].rglob("*.tif"))
        self.assertEqual(len(names), 7)
        for name in names:
            a, b = imageio.imread(outputs[0] / name), imageio.imread(outputs[1] / name)
            self.assertEqual(a.shape, (7, 7))
            self.assertEqual(a.dtype, np.float32)
            np.testing.assert_array_equal(a, b)

    def test_single_mesh_slice_matches_on_rectangular_detector(self):
        slices = []
        for suffix in ("", "_cuda"):
            filename = self.path / ("slice" + suffix + ".tif")
            self.run_example(
                "mesh" + suffix,
                [
                    "--input",
                    self.source,
                    "--n-y",
                    19,
                    "--n-x",
                    27,
                    "--pixel-size",
                    0.13,
                    "--x-rotate",
                    17,
                    "--compute-slice",
                    "--slice-filename",
                    filename,
                    "--no-show",
                ],
            )
            slices.append(imageio.imread(filename))
        self.assertEqual(slices[0].shape, (19, 27))
        self.assertGreater(slices[0].max(), 0)
        np.testing.assert_array_equal(*slices)

    def test_cuda_conebeam_distances_have_mesh_units(self):
        prefix = self.path / "cone"
        self.run_example(
            "mesh_cuda",
            [
                "--n",
                23,
                "--mesh-units",
                "mm",
                "--pixel-size",
                200,
                "--pixel-size-units",
                "um",
                "--sod",
                10,
                "--sdd",
                20,
                "--conebeam",
                "--subsampling",
                1,
                "--use-normals",
                "--double-precision",
                "--projection-filename",
                prefix,
                "--no-show",
            ],
        )
        projection = imageio.imread(str(prefix) + "_00000.tif")
        self.assertTrue(np.isfinite(projection).all())
        self.assertEqual(projection.dtype, np.float64)
        self.assertAlmostEqual(projection[11, 11], 2e-3, delta=5e-8)


if __name__ == "__main__":
    unittest.main()
