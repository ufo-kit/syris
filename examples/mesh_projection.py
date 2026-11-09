"""Common geometry, sampling and output conventions for the mesh examples."""

import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import quantities as q

import syris
import syris.config as cfg
from syris.bodies.mesh import Mesh
from syris.devices.cameras import Camera
from syris.geometry import Trajectory

from .util import add_cuda_mesh_options, validate_cuda_mesh_options

LOG = logging.getLogger(__name__)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("must be a positive integer")
    return value


def add_mesh_options(parser):
    add_cuda_mesh_options(parser)
    parser.add_argument("--mesh-units", "--units", default="um", help="Mesh coordinate units")
    parser.add_argument("--center", choices=["bbox", "gravity", "none"], default="bbox")
    parser.add_argument("--device-index", type=int, default=0, help="Compute device index")
    parser.add_argument("--platform-name", help="OpenCL platform (defaults to NVIDIA if available)")
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed projections before each scan"
    )
    parser.add_argument("--conebeam", action="store_true", help="CUDA only: cone-beam geometry")
    parser.add_argument("--sod", type=float, help="CUDA source-to-object distance in mesh units")
    parser.add_argument("--sdd", type=float, help="CUDA source-to-detector distance in mesh units")
    parser.add_argument(
        "--subsampling",
        "--adaptive-depth",
        type=int,
        default=0,
        help="CUDA only: adaptive refinement depth; 0 uses pixel centers",
    )
    parser.add_argument(
        "--use-normals",
        action="store_true",
        help="CUDA only: oriented material boundaries instead of paired intersections",
    )
    parser.add_argument(
        "--supersampling-projection",
        type=int,
        choices=range(1, 5),
        default=1,
        help="OpenCL only: legacy median sampling factor; keep 1 for backend comparisons",
    )


def validate_mesh_options(parser, args, backend):
    validate_cuda_mesh_options(parser, args)
    if args.device_index < 0 or args.warmup < 0:
        parser.error("--device-index and --warmup must be nonnegative")
    max_depth = 2 if args.double_precision else 3
    if not 0 <= args.subsampling <= max_depth:
        parser.error(f"--subsampling must be between 0 and {max_depth} for this precision")
    if backend == "opencl" and (
        args.conebeam or args.subsampling or args.mixed_precision or args.use_normals
    ):
        parser.error("--conebeam, --subsampling, --mixed-precision and --use-normals require CUDA")
    if backend == "cuda" and args.supersampling_projection != 1:
        parser.error(
            "--supersampling-projection is OpenCL-only; use --supersampling on both backends"
        )
    if not args.conebeam and (args.sod is not None or args.sdd is not None):
        parser.error("--sod and --sdd require --conebeam")
    if any(value is not None and value <= 0 for value in (args.sod, args.sdd)):
        parser.error("--sod and --sdd must be positive")
    if args.sod is not None and args.sdd is not None and args.sdd <= args.sod:
        parser.error("--sdd must be greater than --sod")
    args.backend = backend
    args.mesh_units = q.Quantity(1, args.mesh_units)
    args.center = None if args.center == "none" else args.center
    for name in ("sod", "sdd"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name) * args.mesh_units)


def initialize(args, device_index=0):
    # Slicing has no CUDA implementation; label it explicitly in both entry points.
    backend = "opencl" if getattr(args, "make_gt", False) else args.backend
    if backend == "cuda":
        import cupy as cp

        cp.cuda.Device(device_index).use()
    syris.init(
        compute_backend=backend,
        platform_name=args.platform_name,
        device_index=device_index,
        logfile=getattr(args, "logfile", None),
        loglevel=logging.INFO,
        double_precision=args.double_precision,
        mixed_precision=args.mixed_precision if backend == "cuda" else False,
    )
    if cfg.BACKEND.name != backend:
        raise RuntimeError(f"Requested {backend}, but syris selected {cfg.BACKEND.name}")
    if backend == "cuda":
        device_name = cp.cuda.runtime.getDeviceProperties(device_index)["name"].decode()
    else:
        device_name = cfg.OPENCL.queue.device.name
    LOG.info(
        "Backend: %s; device %s: %s; dtype: %s",
        backend,
        device_index,
        device_name,
        np.dtype(cfg.PRECISION.np_float),
    )
    if getattr(args, "make_gt", False):
        LOG.info("Ground-truth slicing uses OpenCL in both examples")


def load_mesh(args):
    """Load exactly the same triangles, units and centering for either backend."""
    source = args.input
    if source is None:
        source = pv.Box(bounds=(-1, 1, -1, 1, -1, 1)).triangulate()
        LOG.info("Using the built-in cube (no download)")
    elif Path(source).suffix.lower() == ".npy":
        # Preserve the original scan example's (3, 3 * triangle_count) input format.
        vertices = np.load(source)
        if vertices.ndim != 2 or vertices.shape[0] != 3 or vertices.shape[1] % 3:
            raise ValueError("A .npy mesh must have shape (3, 3 * triangle_count)")
        faces = np.column_stack(
            (np.full(vertices.shape[1] // 3, 3), np.arange(vertices.shape[1]).reshape(-1, 3))
        )
        source = pv.PolyData(vertices.T, faces)
    mesh = Mesh.from_file(
        source,
        Trajectory([(0, 0, 0)] * args.mesh_units),
        unit=args.mesh_units,
        center=args.center,
        iterations=args.supersampling_projection,
        use_normals=args.use_normals,
        repair=args.repair_mesh,
        float32_rounding=args.float32_rounding,
    )
    LOG.info("Number of triangles: %s", mesh.num_triangles)
    return mesh


def rotation_span(mesh):
    """Diameter about the actual rotation origin, including off-center inputs."""
    return 2 * np.max(np.linalg.norm(mesh.triangles.magnitude, axis=0)) * cfg.UNIT


def cone_distances(args, span):
    sod = 2 * span if args.sod is None else args.sod
    sdd = sod + 1.5 * span if args.sdd is None else args.sdd
    if sdd <= sod:
        raise ValueError("Source-to-detector distance must exceed source-to-object distance")
    return sod, sdd


class Projector:
    """Project fixed detector rays; return a host array of thickness in metres.

    Both backends trace one ray at each fine-pixel center, then use the same
    NumPy area average. Backend-specific sampling modes are explicit opt-ins.
    """

    def __init__(self, mesh, shape, pixel_size, args):
        self.mesh = mesh
        self.backend = args.backend
        self.shape = tuple(shape)
        self.ss = args.supersampling
        self.fine_shape = tuple(n * self.ss for n in shape)
        self.pixel_size = pixel_size / self.ss
        self.conebeam = args.conebeam
        self.depth = args.subsampling
        self.camera = None
        if self.backend == "cuda":
            span = rotation_span(mesh)
            if args.conebeam:
                sod, sdd = cone_distances(args, span)
                distance = sdd - sod
            else:
                distance, sdd = span, 0 * cfg.UNIT
            self.camera = Camera(
                shape=np.array(self.fine_shape),
                pixel_size=self.pixel_size,
                source_detector_distance=sdd,
            )
            # OpenCL's first pixel center is (ps/2, ps/2), with rays along +z.
            center = np.array([shape[1] / 2, shape[0] / 2, 0]) * pixel_size
            center[2] = -distance
            self.camera.translate(center)

    def project(self):
        if self.backend == "cuda":
            projection = self.mesh.project(
                camera=self.camera,
                parallel=not self.conebeam,
                iterations=self.depth,
            ).get()
            # OpenCL already returns metres; CUDA returns cfg.UNIT magnitudes.
            projection *= float(cfg.UNIT.rescale(q.m).magnitude)
        else:
            projection = self.mesh.project(self.fine_shape, self.pixel_size, t=None).get()
        if self.ss > 1:
            projection = projection.reshape(self.shape[0], self.ss, self.shape[1], self.ss)
            projection = projection.mean(axis=(1, 3))
        return projection


def detector_center(shape, pixel_size):
    return np.array([shape[1] / 2, shape[0] / 2, 0]) * pixel_size


def log_projection_setup(args, shape, pixel_size):
    LOG.info(
        "Image: %s; pixel size: %s; uniform samples/pixel: %s; output thickness: m",
        shape,
        pixel_size.rescale(q.um),
        args.supersampling**2,
    )
    if (
        args.subsampling
        or args.supersampling_projection != 1
        or args.use_normals
        or args.mixed_precision
        or args.conebeam
    ):
        LOG.info("Backend-specific mode enabled; this is not a matched OpenCL/CUDA workload")
