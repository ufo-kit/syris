"""Profile warmed projections with Nsight Systems or Nsight Compute.

Run with python -m benchmarks.profile_cuda_projection. CUDA profiler start/stop
excludes initialization, compilation, BVH construction, and warmup. NVTX ranges
identify each depth and repeat. Profiling timings are not benchmark timings.
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import cupy as cp
import numpy as np
import quantities as q
from cupyx.profiler import profile, time_range

import syris
from syris.bodies.mesh import Mesh
from syris.devices.cameras import Camera
from syris.geometry import Y_AX, Trajectory
from syris.gpu.cuda_utils import CudaPipeline

from .cuda_projection import load_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--precision", choices=("float32", "float64", "mixed"), default="float32")
    parser.add_argument("--camera", choices=("parallel", "cone"), default="parallel")
    parser.add_argument("--angle", type=float, default=37)
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--line-info",
        action="store_true",
        help="Compile the projection module with -lineinfo for source correlation",
    )
    parser.add_argument("--warp-tile-width", type=int, choices=(0, 1, 2, 4, 8, 16, 32), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    max_depth = 2 if args.precision == "float64" else 3
    if min(args.size, args.repeats, args.warmup) < 1:
        parser.error("size, repeats, and warmup must be positive")
    if any(depth < 0 or depth > max_depth for depth in args.depths):
        parser.error(f"depths must be between 0 and {max_depth}")
    if not np.isfinite(args.angle):
        parser.error("angle must be finite")

    print(f"Initializing {args.precision} CUDA projection", flush=True)
    syris.init(
        compute_backend="cuda",
        double_precision=args.precision == "float64",
        mixed_precision=args.precision == "mixed",
        warp_tile_width=args.warp_tile_width,
        loglevel=logging.WARNING,
    )
    data = load_mesh(args.mesh).triangulate()
    faces = data.faces.reshape(-1, 4)[:, 1:]
    vertices = np.asarray(data.points, np.float64)[faces]
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    data.cell_data["Normals"] = normals
    mesh = Mesh.from_file(
        data, Trajectory([(0, 0, 0)] * q.um), center=None, unit=q.um, use_normals=True
    )
    print(f"Building BVH: {mesh.num_triangles} triangles", flush=True)
    accel = mesh.build_accelerator()
    if args.line_info:
        options = (
            ["-D__FP_T_D__"]
            if args.precision == "float64"
            else ["-DSYRIS_MIXED_PRECISION"]
            if args.precision == "mixed"
            else []
        )
        if args.warp_tile_width:
            options.append(f"-DSYRIS_WARP_TILE_WIDTH={args.warp_tile_width}")
        print("Compiling projection module with line information", flush=True)
        source = Path(syris.__file__).parent / "gpu" / "cuda"
        pipeline = CudaPipeline([str(source)], options + ["-lineinfo"])
        # Match ordinary initialization's source set; only add line metadata.
        pipeline.readModuleFromFiles(
            "profile",
            [source / name for name in ("WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu")],
        )
        for name in ("project_parallel_normals_kernel", "project_conebeam_normals_kernel"):
            pipeline.getKernelFromModule("profile", name)
        accel.pipeline = pipeline

    xyz = np.asarray(mesh._current[:3])
    lo, hi = xyz.min(axis=1), xyz.max(axis=1)
    center, span = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
    parallel = args.camera == "parallel"
    camera = Camera(
        pixel_size=span * 1.2 * (1 if parallel else 1.75) / args.size * q.um,
        shape=(args.size, args.size),
        trajectory=Trajectory([(0, 0, 0)] * q.um),
        source_detector_distance=(0 if parallel else 3.5 * span) * q.um,
    )
    camera.translate((center - [0, 0, span * (1 if parallel else 1.5)]) * q.um)
    camera.rotate_around(args.angle * q.deg, Y_AX, pivot=center * q.um)

    def project(depth):
        return accel.project(camera=camera, parallel=parallel, iterations=depth)

    reference = {}
    for depth in args.depths:
        print(f"Warming depth {depth}", flush=True)
        for _ in range(args.warmup):
            reference[depth] = project(depth)
        cp.cuda.get_current_stream().synchronize()
    # All downloads and correctness comparisons stay outside the capture.
    reference = {depth: value.get() for depth, value in reference.items()}
    captured, timings = {}, []
    print("Starting warmed capture", flush=True)
    with profile():
        for depth in args.depths:
            for repeat in range(args.repeats):
                label = (
                    f"projection/{args.precision}/{args.camera}/tile={args.warp_tile_width}"
                    f"/depth={depth}/repeat={repeat}"
                )
                start = time.perf_counter()
                with time_range(label):
                    captured[depth] = project(depth)
                    cp.cuda.get_current_stream().synchronize()
                timings.append(
                    {
                        "depth": depth,
                        "repeat": repeat,
                        "instrumented_wall_ms": (time.perf_counter() - start) * 1000,
                    }
                )
    print("Capture complete; checking outputs", flush=True)
    comparisons = {}
    for depth, value in captured.items():
        actual = value.get()
        comparisons[depth] = {
            "nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
            "changed_pixels": int(np.count_nonzero(actual != reference[depth])),
        }
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    source = Path(syris.__file__).parent / "gpu" / "cuda"
    report = {
        "gpu": props["name"].decode(),
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "mesh": str(args.mesh.resolve()),
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "triangles": mesh.num_triangles,
        "size": args.size,
        "precision": args.precision,
        "camera": args.camera,
        "angle": args.angle,
        "depths": args.depths,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "line_info": args.line_info,
        "warp_tile_width": args.warp_tile_width,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.iterdir())
            if p.is_file()
        },
        "timing_note": "Instrumented wall times, not performance benchmarks. Capture excludes setup and output download.",
        "timings": timings,
        "comparisons": comparisons,
        "kernel_attributes": accel.pipeline.kernels[
            f"project_{'parallel' if parallel else 'conebeam'}_normals_kernel"
        ].attributes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(result["changed_pixels"] or result["nonfinite"] for result in comparisons.values()):
        raise AssertionError(f"Projection differs from warmup or is nonfinite; see {args.output}")
    print(f"Saved {args.output}; all compared outputs match warmup exactly", flush=True)


if __name__ == "__main__":
    main()
