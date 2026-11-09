"""Controlled CUDA rounding experiments; never changes production sources."""

import argparse
import hashlib
import json
import math
import time
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import quantities as q
import tifffile

from examples.mesh_projection import Projector, detector_center
from syris import config as cfg
from syris.bodies.accelerators import BvhCupyAccelerator
from syris.bodies.mesh import Mesh
from syris.bodies.meshreader import PyvistaReader
from syris.geometry import Y_AX, Trajectory
from syris.gpu.cuda_utils import CudaPipeline
from syris.tests.cuda_projection_util import ProjectionHarness

REPO = Path(__file__).resolve().parents[2]
BENCH = None


def make_mesh(points, faces, use_normals=False):
    poly = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)))
    reader = PyvistaReader(
        poly, unit=q.um, dtype=cfg.PRECISION.np_float, repair=False, float32_rounding=False
    )
    return Mesh(
        reader.vertices,
        Trajectory([(0, 0, 0)] * q.um),
        center=None,
        iterations=1,
        bounds=reader.bounds,
        epsilon=reader.epsilon,
        normals=reader.normals if use_normals else None,
        use_normals=use_normals,
        dynamic_range=reader.dynamic_range,
    )


class ScanProjector:
    """Reproduce the benchmark's rotating-mesh pose and shared example camera."""

    def __init__(self, mesh, fov_um, n, config, backend, workflow):
        if backend != "cuda" or workflow != "rotating":
            raise ValueError("This experiment covers rotating-mesh CUDA scans only")
        self.mesh = mesh
        args = SimpleNamespace(
            backend=backend,
            supersampling=config.supersampling,
            conebeam=False,
            subsampling=config.cuda_adaptive_depth,
        )
        self.projector = Projector(mesh, (n, n), fov_um / n * q.um, args)
        self.center = detector_center((n, n), fov_um / n * q.um)

    def pose(self, angle):
        self.mesh.clear_transformation()
        self.mesh.translate(self.center)
        self.mesh.rotate(angle * q.deg, Y_AX)


def add_path_arguments(parser):
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Existing benchmark repository containing prepared meshes and audited TIFFs",
    )
    parser.add_argument(
        "--work-directory",
        type=Path,
        default=ROOT,
        help="Scratch directory for isolated CUDA source copies",
    )


def configure_paths(args):
    if (REPO / "syris/gpu/cuda/Detector.cuh").exists():
        raise SystemExit(
            "These historical source-patching experiments require the pre-integration "
            "SYRIS sources recorded in the result hashes. Use "
            "benchmarks.rounding_precision.public_modes for the current implementation."
        )
    global BENCH, ROOT
    BENCH = args.benchmark_root.resolve()
    ROOT = args.work_directory.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)


def provenance():
    import datetime
    import platform
    import subprocess
    import sys

    import cupy as cp

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    files = list((REPO / "syris/gpu/cuda").glob("*.cu")) + list(
        (REPO / "syris/gpu/cuda").glob("*.cuh")
    )
    files += [
        REPO / name
        for name in [
            "syris/config.py",
            "syris/bodies/accelerators.py",
            "syris/bodies/mesh.py",
            "syris/devices/cameras.py",
            "examples/mesh_projection.py",
        ]
    ]
    files += list(Path(__file__).parent.glob("*.py"))
    inputs = [
        BENCH / f"data/prepared/{mesh}/float32-baseline.npz" for mesh in ["bunny", "armadillo"]
    ]
    inputs += [BENCH / f"data/baseline/{mesh}.json" for mesh in ["bunny", "armadillo"]]
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_sha256": {str(path.relative_to(REPO)): digest(path) for path in files},
        "input_sha256": {str(path.relative_to(BENCH)): digest(path) for path in inputs},
        "geometry_contract": "Every precision uses the same stored Float32 geometry; Float64 promotes those coordinates exactly.",
        "scope": "Parallel beam, one sample per pixel; experimental source copies and coordinate changes, no production edits.",
    }


ROOT = Path("/tmp/syris-rounding-experiments")
PIXELS = [(636, 578), (989, 1485), (1165, 1304), (1395, 603)]

CAPTURE = r"""
extern "C" __global__ void capture_rounding_rays(
  int count, const uint2* pixels, uint2 N, FP_T4 U, FP_T4 V, FP_T4 W,
  FP_T4 upperleft_origin, FP_T2 ps, FP_T4* origins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned row = pixels[i].x, col = pixels[i].y;
    FP_T4 scaled_U = U * ps.x, scaled_V = V * ps.y;
    FP_T4 base_pixel_origin = upperleft_origin + scaled_U * col + scaled_V * row;
    FP_T4 sample_origin = base_pixel_origin + scaled_U * 0.5 + scaled_V * 0.5;
    origins[i] = sample_origin;
}
"""
OLD_RAY = "FP_T4 sample_origin = base_pixel_origin + scaled_U * 0.5 + scaled_V * 0.5;"
NEW_RAY = """FP_T4 center_origin = upperleft_origin + scaled_U * (FP_T(N.x) * FP_T(0.5)) + scaled_V * (FP_T(N.y) * FP_T(0.5));
            FP_T4 sample_origin = center_origin + scaled_U * ((FP_T(col) + FP_T(0.5)) - FP_T(N.x) * FP_T(0.5)) + scaled_V * ((FP_T(row) + FP_T(0.5)) - FP_T(N.y) * FP_T(0.5));"""


class CapturePipeline:
    def __init__(self, inner):
        self.inner = inner
        self.last_args = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def launchKernel(self, name, *args):
        if name == "project_parallel_kernel":
            self.last_args = args[2]
        return self.inner.launchKernel(name, *args)


def pipeline_for(harness, ray_mode):
    if ray_mode == "current":
        directory = REPO / "syris/gpu/cuda"
    else:
        directory = ROOT / "sources" / ray_mode
        directory.mkdir(parents=True, exist_ok=True)
        for p in (REPO / "syris/gpu/cuda").iterdir():
            if p.suffix in (".cu", ".cuh"):
                text = p.read_text()
                if p.name == "source.cu":
                    # Only the single-sample parallel projectors; adaptive/cone unmodified.
                    start = text.index('extern "C" __global__ void project_parallel_kernel(')
                    end = text.index('extern "C" __global__ void project_conebeam_kernel(')
                    middle = text[start:end]
                    assert middle.count(OLD_RAY) == 2
                    text = text[:start] + middle.replace(OLD_RAY, NEW_RAY) + text[end:]
                (directory / p.name).write_text(text)
    probe = ROOT / ("capture_" + ray_mode + ".cu")
    probe.write_text(CAPTURE if ray_mode == "current" else CAPTURE.replace(OLD_RAY, NEW_RAY))
    options = ["-D__FP_T_D__"] if harness.dtype is np.float64 else []
    if harness.mixed:
        options.append("-DSYRIS_MIXED_PRECISION")
    result = CudaPipeline([str(directory)], options)
    result.readModuleFromFiles(
        "rounding",
        [directory / n for n in ("WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu")] + [probe],
    )
    for name in [
        "projectTriangleCentroid",
        "growTreeKernel",
        "project_parallel_kernel",
        "capture_rounding_rays",
    ]:
        result.getKernelFromModule("rounding", name)
    return result


def decimal_reference(points, faces, xy):
    with localcontext() as ctx:
        ctx.prec = 90
        v = [[Decimal.from_float(float(c)) for c in p] for p in points]
        x, y = [c if isinstance(c, Decimal) else Decimal.from_float(float(c)) for c in xy]
        hits = []
        for i, face in enumerate(faces):
            a, b, c = [v[j] for j in face]
            e, f = [[p[k] - a[k] for k in range(3)] for p in (b, c)]
            normal = [
                e[1] * f[2] - e[2] * f[1],
                e[2] * f[0] - e[0] * f[2],
                e[0] * f[1] - e[1] * f[0],
            ]
            if normal[2] == 0:
                continue
            signs = [
                (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
                for a, b in [(a, b), (b, c), (c, a)]
            ]
            if not (all(s >= 0 for s in signs) or all(s <= 0 for s in signs)):
                continue
            if any(s == 0 for s in signs):
                raise ValueError("Exact edge excluded from decimal reference")
            z = a[2] - (normal[0] * (x - a[0]) + normal[1] * (y - a[1])) / normal[2]
            hits.append(z)
        hits.sort()
        if len(hits) % 2:
            raise ValueError("Odd reference crossing count")
        return float(sum((hits[i + 1] - hits[i] for i in range(0, len(hits), 2)), Decimal(0)))


def main():
    ap = argparse.ArgumentParser()
    add_path_arguments(ap)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--variants", nargs="*")
    args = ap.parse_args()
    configure_paths(args)
    args.output = args.output or ROOT / "initial.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import cupy as cp

    with np.load(BENCH / "data/prepared/bunny/float32-baseline.npz") as data:
        points, faces = data["points"], data["faces"]
    fov = json.loads((BENCH / "data/baseline/bunny.json").read_text())["fov_um"]
    audit = json.loads((BENCH / ".audit/cuda-counterexamples/audit.json").read_text())
    refs = [r["reference_um"] for r in audit["rows"] if r["mesh"] == "bunny"]
    results = []
    metadata = provenance()
    gpu = {}
    for mode in ["float32", "mixed", "float64"]:
        print("Compile", mode, flush=True)
        h = ProjectionHarness(double=mode == "float64", mixed_precision=mode == "mixed")
        h.mixed = mode == "mixed"
        gpu[mode] = h
    variants = [
        (m, c, s, r)
        for m in ["float32", "mixed"]
        for c, s, r in [
            (False, "current", "current"),
            (False, "power2", "current"),
            (True, "power2", "current"),
            (True, "power2", "centered"),
            (True, "none", "centered"),
        ]
    ]
    variants += [("float64", False, "current", "current")]
    compiled = {}
    for mode, centered, scaling, rays in variants:
        name = f"{mode}-{'centered' if centered else 'positive'}-{scaling}-{rays}"
        if args.variants and name not in args.variants:
            continue
        print("Run", name, flush=True)
        h = gpu[mode]
        key = (mode, rays)
        if key not in compiled:
            compiled[key] = pipeline_for(h, rays)
        h.pipeline = CapturePipeline(compiled[key])
        with h.configured():
            mesh = make_mesh(points, faces)
            scan = ScanProjector(
                mesh,
                fov,
                2048,
                SimpleNamespace(supersampling=1, cuda_adaptive_depth=0),
                "cuda",
                "rotating",
            )
            scan.pose(0)
            camera = scan.projector.camera
            if centered:
                mesh.clear_transformation()
                distance = float(camera.position.rescale(q.um).magnitude[2])
                camera.clear_transformation()
                camera.translate(np.array([0.0, 0.0, distance]) * q.um)
            if scaling == "power2":
                mesh._furthest_point = 2.0 ** math.ceil(math.log2(mesh._furthest_point))
            elif scaling == "none":
                mesh._furthest_point = 1.0
            accel = BvhCupyAccelerator(mesh)
            accel.build()
            scale = 1 / float(mesh.furthest_point.magnitude)
            times = []
            for rep in range(4):
                cp.cuda.get_current_stream().synchronize()
                start = time.perf_counter()
                frame = accel.project(camera=camera, parallel=True, iterations=0).get()
                frame *= 1e-6
                cp.cuda.get_current_stream().synchronize()
                if rep:
                    times.append(time.perf_counter() - start)
            a = h.pipeline.last_args
            origins = cp.empty((len(PIXELS), 4), h.dtype)
            compiled[key].kernels["capture_rounding_rays"](
                (1,),
                (32,),
                (np.int32(len(PIXELS)), cp.asarray(PIXELS, np.uint32), *a[4:10], origins),
            )
            origins = origins.get()
            vertices = a[3].view(h.dtype).reshape(-1, 4).get()[:, :3]
            soupfaces = np.arange(len(vertices)).reshape(-1, 3)
            encoded = [decimal_reference(vertices, soupfaces, o[:2]) / scale for o in origins]
            values = [float(frame[p]) * 1e6 for p in PIXELS]
            record = dict(
                variant=name,
                scale=scale,
                values_um=values,
                errors_nm=[(v - r) * 1e3 for v, r in zip(values, refs)],
                encoded_input_error_nm=[(v - r) * 1e3 for v, r in zip(encoded, refs)],
                remaining_arithmetic_error_nm=[(v - e) * 1e3 for v, e in zip(values, encoded)],
                projection_host_ms=[t * 1e3 for t in times],
                nonfinite=int(np.sum(~np.isfinite(frame))),
                negative=int(np.sum(frame < 0)),
            )
            if name == "float32-positive-current-current":
                old = tifffile.imread(
                    BENCH
                    / "results/two-day/originals/runs/bunny-rotating-2048-float32-baseline-cuda-0/representatives/000000.tif"
                )
                record["bitwise_equal_retained_tiff"] = bool(
                    np.array_equal(frame, old, equal_nan=True)
                )
                if not record["bitwise_equal_retained_tiff"]:
                    raise RuntimeError("Current baseline differs from the retained benchmark TIFF")
            results.append(record)
            Path(args.output).write_text(
                json.dumps(
                    dict(provenance=metadata, pixels=PIXELS, reference_um=refs, results=results),
                    indent=2,
                )
            )
            print(json.dumps(record), flush=True)
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
