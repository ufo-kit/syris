"""CUDA test fixtures using the production loader, BVH builder, and projector."""

import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import quantities as q


def cuda_or_skip():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA device")
    except (ImportError, RuntimeError) as exc:
        if os.environ.get("SYRIS_REQUIRE_CUDA") == "1":
            raise RuntimeError("CUDA is required for this test run") from exc
        raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc
    return cp


def box(x0=-0.5, x1=0.5, y0=-0.5, y1=0.5, z0=0.0, z1=1.0):
    """Outward-oriented closed box, with two triangles on every face."""
    points = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = [
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
    ]
    return points[faces]


class ProjectionHarness:
    def __init__(
        self,
        double=False,
        mixed_precision=False,
        hit_capacity=None,
        signed_thickness=False,
        box_scalar_selection=None,
        warp_tile_width=None,
    ):
        if double and mixed_precision:
            raise ValueError("Choose double or mixed precision, not both.")
        self.cp = cuda_or_skip()
        from syris import config as cfg
        from syris.gpu.cuda_utils import CudaPipeline

        self.precision = cfg.Precision(double)
        self.dtype = self.precision.np_float
        kernel_dir = Path(__file__).resolve().parents[1] / "gpu" / "cuda"
        options = ["-D__FP_T_D__"] if double else []
        if mixed_precision:
            options.append("-DSYRIS_MIXED_PRECISION")
        if hit_capacity is not None:
            options.append(f"-DSYRIS_HIT_CAPACITY={hit_capacity}")
        if signed_thickness:
            options.append("-DSYRIS_SIGNED_THICKNESS")
        if box_scalar_selection is not None:
            options.append(f"-DSYRIS_BOX_SCALAR_SELECTION={int(box_scalar_selection)}")
        if warp_tile_width is not None:
            options.append(f"-DSYRIS_WARP_TILE_WIDTH={warp_tile_width}")
        self.pipeline = CudaPipeline([str(kernel_dir)], options)
        sources = [kernel_dir / f for f in ("WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu")]
        sources.append(Path(__file__).parent / "cuda" / "projection_probe.cu")
        self.pipeline.readModuleFromFiles("tests", sources)
        names = [
            "test_triangles",
            "test_boxes",
            "test_trace",
            "projectTriangleCentroid",
            "growTreeKernel",
            "project_parallel_kernel",
            "project_parallel_normals_kernel",
            "project_conebeam_kernel",
            "project_conebeam_normals_kernel",
            "project_shared_samples_kernel",
            "project_missing_samples_kernel",
            "classify_shared_samples_kernel",
            "refine_shared_samples_kernel",
        ]
        for name in names:
            self.pipeline.getKernelFromModule("tests", name)
        rc = cfg.RayCasting(double, 1)
        self.eps = self.cp.asarray(
            [
                getattr(rc, "p_" + name)
                for name in (
                    "ray_box_epsilon",
                    "tri_ray_tmin",
                    "tri_gamma_multiplier",
                    "tri_abs_min_error",
                    "tri_d_gamma_multiplier",
                    "tri_d_abs_min_error",
                    "group_abs_epsilon",
                    "group_rel_epsilon",
                    "unique_abs_epsilon",
                )
            ],
            dtype=self.dtype,
        )

    @contextmanager
    def configured(self):
        from syris import config as cfg

        old = {name: getattr(cfg, name, None) for name in ("BACKEND", "PRECISION", "UNIT")}
        cfg.BACKEND = SimpleNamespace(xp=self.cp, pipeline=self.pipeline)
        cfg.PRECISION = self.precision
        cfg.UNIT = q.um
        try:
            yield
        finally:
            for name, value in old.items():
                setattr(cfg, name, value)

    def vectors(self, value):
        value = np.asarray(value, dtype=self.dtype)
        return self.cp.asarray(np.pad(value, [(0, 0)] * (value.ndim - 1) + [(0, 1)]))

    def launch(self, name, n, args):
        self.pipeline.kernels[name](((n + 127) // 128,), (128,), args)

    def triangles(self, origins, directions, vertices):
        n = len(origins)
        hits = self.cp.empty((n, 2), np.int32)
        distances = self.cp.empty(n, self.dtype)
        self.launch(
            "test_triangles",
            n,
            (
                np.int32(n),
                self.vectors(origins),
                self.vectors(directions),
                self.vectors(vertices).reshape(-1, 4),
                self.eps,
                hits,
                distances,
            ),
        )
        return hits.get(), distances.get()

    def boxes(self, origins, directions, lower, upper, ranges):
        n = len(origins)
        hits = self.cp.empty(n, np.int32)
        self.launch(
            "test_boxes",
            n,
            (
                np.int32(n),
                self.vectors(origins),
                self.vectors(directions),
                self.vectors(lower),
                self.vectors(upper),
                self.cp.asarray(ranges, dtype=self.dtype),
                self.eps,
                hits,
            ),
        )
        return hits.get()

    def build(self, vertices, use_normals=True):
        from syris.bodies.accelerators import BvhCupyAccelerator

        vertices = np.asarray(vertices, dtype=self.dtype).reshape(-1, 3, 3)
        lo = vertices.min(axis=(0, 1))
        hi = vertices.max(axis=(0, 1))
        mesh = SimpleNamespace(
            transform=lambda: None,
            _current=np.vstack((vertices.reshape(-1, 3).T, np.ones(3 * len(vertices)))),
            num_triangles=len(vertices),
            normals=np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            _use_normals=use_normals,
            bounds=np.column_stack([lo, hi]).ravel(),
            epsilon=0,
            dynamic_range=1,
            _state=0,
            furthest_point=1 * q.um,
        )
        with self.configured():
            accel = BvhCupyAccelerator(mesh)
            accel.build()
        return accel

    def trace(self, accel, origins, directions):
        t = accel.tree
        n = len(origins)
        out = self.cp.empty((n, 2), self.dtype)
        self.launch(
            "test_trace",
            n,
            (
                np.int32(n),
                np.uint32(len(t["keys"])),
                self.vectors(origins),
                self.vectors(directions),
                t["vertices"],
                t["normals"],
                t["indices"],
                t["rope"],
                t["left"],
                t["bbMin"],
                t["bbMax"],
                self.eps,
                out,
            ),
        )
        return out.get()

    def project(self, accel, **kwargs):
        return float(self.project_image(accel, **kwargs)[0, 0])

    def project_image(
        self,
        accel,
        *,
        parallel=True,
        depth=0,
        origin=(0, 0, -1),
        pixel_size=(1, 1),
        source=(0.5, 0.5, -3),
        shape=(1, 1),
        basis=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        **kwargs,
    ):
        def vec(x):
            return np.array([*x, 0], dtype=self.dtype)

        camera = SimpleNamespace(
            shape=shape,
            viewport_basis_vectors=tuple(vec(axis) for axis in basis),
            kernel_shape_xy=np.array(shape[::-1], dtype=np.uint32),
            p00_corner=vec(origin),
            kernel_pixel_size_xy=np.array(pixel_size, dtype=self.dtype),
            source_point=vec(source),
        )
        with self.configured():
            return accel.project(camera=camera, parallel=parallel, iterations=depth, **kwargs).get()
