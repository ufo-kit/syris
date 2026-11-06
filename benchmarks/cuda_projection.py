"""Warmed CUDA projection ablations; run from the checkout with python -m.

Example:
  python -m benchmarks.cuda_projection --mesh path/to/mesh.ply --size 2048 \
      --depths 0 1 --output /tmp/cuda-projection.json

The reference CUDA sources are extracted from git, never edited in place.
No mesh is downloaded or repaired by this benchmark. NPZ input may contain
indexed ``vertices`` and ``faces`` arrays, e.g. a previously repaired mesh.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time

import cupy as cp
import numpy as np
import pyvista as pv
import quantities as q

import syris
from syris import config as cfg
from syris.bodies.mesh import Mesh
from syris.devices.cameras import Camera
from syris.geometry import Trajectory, Y_AX
from syris.gpu.cuda_utils import CudaPipeline


PROJECTORS = [f"project_{mode}{normals}_kernel"
              for mode in ("parallel", "conebeam") for normals in ("", "_normals")]
STAGES = ["project_shared_samples_kernel", "project_missing_samples_kernel", "classify_shared_samples_kernel",
          "refine_shared_samples_kernel"]


class MeasuredPipeline:
    """Measure on the stream that actually launches kernels, excluding setup."""

    def __init__(self, pipeline, full_grid=False, block_size=None, worker_blocks=None):
        self.pipeline = pipeline
        self.full_grid = full_grid
        self.block_size = block_size
        self.worker_blocks = worker_blocks
        self.kernel_ms = 0
        self.stage_ms = {}

    def synchronize(self):
        cp.cuda.get_current_stream().synchronize()

    def launchKernel(self, name, *args):
        if self.block_size is not None and name in PROJECTORS:
            _, _, kernel_args = args
            args = (((kernel_args[2].size + self.block_size - 1) // self.block_size, 1, 1),
                    (self.block_size, 1, 1), kernel_args)
        if self.full_grid and name in PROJECTORS:
            _, block, kernel_args = args
            args = (((kernel_args[2].size + block[0] - 1) // block[0], 1, 1), block, kernel_args)
        if self.worker_blocks is not None and name in PROJECTORS:
            grid, block, kernel_args = args
            args = ((min(grid[0], self.worker_blocks), 1, 1), block, kernel_args)
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        result = self.pipeline.launchKernel(name, *args)
        end.record()
        end.synchronize()
        elapsed = cp.cuda.get_elapsed_time(start, end)
        self.kernel_ms += elapsed
        self.stage_ms[name] = self.stage_ms.get(name, 0) + elapsed
        return result


def reference_pipeline(ref, directory, options, uninitialized=False):
    directory.mkdir()
    source_root = "syris/gpu/cuda/"
    names = subprocess.check_output(
        ["git", "ls-tree", "--name-only", ref + ":" + source_root], text=True
    ).splitlines()
    for name in names:
        if not name.endswith((".cu", ".cuh")):
            continue
        data = subprocess.check_output(["git", "show", ref + ":" + source_root + name])
        if uninitialized and name == "Commons.cuh":
            old = b"HitRecord() : t(0.0), triangle_idx(0) {}"
            if old not in data:
                raise ValueError("The uninitialized ablation requires the original HitRecord constructor")
            data = data.replace(old, b"HitRecord() = default;")
        (directory / name).write_bytes(data)
    pipeline = CudaPipeline([str(directory)], options.copy())
    pipeline.readModuleFromFiles("reference", [directory / name for name in
                                             ("WatertightRay.cu", "Ray.cu", "source.cu")])
    for name in PROJECTORS:
        pipeline.getKernelFromModule("reference", name)
    return pipeline


def compile_variant(options, flags, names=PROJECTORS):
    source = Path(syris.__file__).parent / "gpu" / "cuda"
    pipeline = CudaPipeline([str(source)], options + flags)
    pipeline.readModuleFromFiles("small_hits", [source / name for name in
                                              ("WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu")])
    for name in names:
        pipeline.getKernelFromModule("small_hits", name)
    return pipeline


class DeferredSignedPipeline:
    """Research-only two-pass signed projection; assumes a 0/1 boundary.

    The fast module emits -inf for exceptional rays/pixels. A device queue
    reprojects exactly those pixels with the ordinary sorted module. The host
    never downloads a count between stages. Only the local sampler is supported.
    """

    def __init__(self, options):
        self.fast = compile_variant(options, ["-DSYRIS_SIGNED_THICKNESS",
                                             "-DSYRIS_SIGNED_DEFER_FALLBACK"],
                                    PROJECTORS + ["queue_signed_fallbacks"])
        self.retry = compile_variant(options, ["-DSYRIS_PROJECTION_RETRY"])
        self.kernels = self.fast.kernels
        self.worker_blocks = 4 * cp.cuda.Device().attributes["MultiProcessorCount"]
        self.last_retry_count = None

    def synchronize(self):
        cp.cuda.get_current_stream().synchronize()

    def launchKernel(self, name, grid, block, args):
        if name not in PROJECTORS or "normals" not in name:
            raise ValueError("Deferred signed projection requires normals and local sampling")
        counter, _, image = args[:3]
        count = cp.zeros(1, cp.uint32)
        indices = cp.empty(image.size, cp.uint32)
        self.last_retry_count = count
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        self.fast.launchKernel(name, grid, block, args)
        self.kernels["queue_signed_fallbacks"](
            ((image.size + 255) // 256,), (256,), (image, np.uint32(image.size), indices, count))
        counter.fill(0)
        self.retry.launchKernel(name, (min(grid[0], self.worker_blocks), 1, 1), block,
                                (*args, indices, count))
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end), None


class DepthSpecializedPipeline:
    """Compile the exact requested depth, without changing its sample pattern."""

    def __init__(self, options, depths, signed=False):
        flags = ["-DSYRIS_SIGNED_THICKNESS"] if signed else []
        self.modules = {depth: compile_variant(options, flags + [f"-DSYRIS_MAX_DEPTH={depth}"])
                        for depth in sorted(set(depths))}
        self.kernels = self.modules[min(self.modules)].kernels

    def synchronize(self):
        cp.cuda.get_current_stream().synchronize()

    def launchKernel(self, name, grid, block, args):
        if name not in PROJECTORS:
            raise ValueError("Depth specialization supports the local projector only")
        depth = int(args[-10])
        if depth not in self.modules:
            raise ValueError(f"No kernel compiled for the requested depth {depth}")
        return self.modules[depth].launchKernel(name, grid, block, args)


def experimental_pipeline(options, variant, depths=()):
    if variant == "scalar_axes" or (variant.startswith("tile") and variant.endswith("_axes")):
        width = int(variant[4:].split("_")[0]) if variant.startswith("tile") else 0
        return compile_variant(options, ["-DSYRIS_BOX_SCALAR_SELECTION=2",
                                         f"-DSYRIS_WARP_TILE_WIDTH={width}"])
    if variant in ("indexed", "scalar"):
        return compile_variant(options, [f"-DSYRIS_BOX_SCALAR_SELECTION={int(variant == 'scalar')}",
                                         "-DSYRIS_WARP_TILE_WIDTH=0"])
    if variant.startswith("tile"):
        width = int(variant[4:].split("_")[0])
        return compile_variant(options, ["-DSYRIS_BOX_SCALAR_SELECTION=1",
                                         f"-DSYRIS_WARP_TILE_WIDTH={width}"])
    if variant == "signed_deferred":
        return DeferredSignedPipeline(options)
    if variant in ("specialized", "signed_specialized"):
        return DepthSpecializedPipeline(options, depths, signed=variant.startswith("signed"))
    flags = ["-DSYRIS_HIT_CAPACITY=32"] if variant == "hybrid" else ["-DSYRIS_SIGNED_THICKNESS"]
    return compile_variant(options, flags)


def layers_mesh(count):
    parts = [pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, i * 0.02, i * 0.02 + 0.01)).triangulate()
             for i in range(count)]
    points = np.concatenate([part.points for part in parts])
    faces = np.concatenate([part.faces.reshape(-1, 4)[:, 1:] + i * 8
                            for i, part in enumerate(parts)])
    return pv.PolyData(points, np.column_stack([np.full(len(faces), 3), faces]).ravel())


def load_mesh(path):
    if path.suffix == ".npz":
        with np.load(path) as data:
            vertices, faces = data["vertices"], data["faces"]
        return pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    return pv.read(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="f75c9af")
    parser.add_argument("--mesh", type=Path, action="append", default=[])
    parser.add_argument("--layers", type=int, nargs="*", default=[20, 80, 140])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--layer-size", type=int, default=32)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--precision", choices=["float32", "float64", "mixed"], default="float32")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "uninitialized", "optimized", "hybrid", "staged", "signed", "signed_deferred", "blocks128", "specialized", "signed_specialized", "indexed", "scalar", "tile4", "tile8", "tile16", "tile32", "tile4_pool", "tile8_pool", "scalar_axes", "tile4_axes", "tile8_axes", "tile16_axes", "tile32_axes"],
                        default=["baseline", "uninitialized", "optimized", "hybrid", "staged"])
    parser.add_argument("--reference", choices=["baseline", "optimized", "indexed", "scalar"], default="baseline")
    parser.add_argument("--assume-material-boundary", action="store_true",
                        help="Required for signed: assert closed, correctly oriented 0/1 material boundaries; not a validation")
    parser.add_argument("--atol", type=float, default=0)
    parser.add_argument("--rtol", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.layer_size, args.repeats) < 1 or any(n < 1 for n in args.layers):
        parser.error("sizes, repeats, and layer counts must be positive")
    if args.reference not in args.variants:
        parser.error("include the reference variant for accuracy comparisons")
    if any(name.startswith("signed") for name in args.variants) and not args.assume_material_boundary:
        parser.error("signed requires --assume-material-boundary; arbitrary meshes are not supported")
    if any(not np.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("comparison tolerances must be finite and nonnegative")
    max_depth = 2 if args.precision == "float64" else 3
    if any(depth < 0 or depth > max_depth for depth in args.depths):
        parser.error(f"depths must be between 0 and {max_depth} in {args.precision}")
    double, mixed = args.precision == "float64", args.precision == "mixed"
    syris.init(compute_backend="cuda", double_precision=double, mixed_precision=mixed,
               loglevel=logging.WARNING)
    options = ["-D__FP_T_D__"] if double else ["-DSYRIS_MIXED_PRECISION"] if mixed else []
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {"baseline_ref": subprocess.check_output(
                  ["git", "rev-parse", args.baseline_ref], text=True).strip(),
              "gpu": props["name"].decode(), "cupy": cp.__version__,
              "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
              "precision": args.precision, "repeats": args.repeats,
              "reference_variant": args.reference, "atol": args.atol, "rtol": args.rtol,
              "comparison_status": "running",
              "assumed_material_boundary": args.assume_material_boundary,
              "timing": "warmed; wall includes preparation and output scaling, excludes download; kernel_ms sums CUDA events on the launch stream",
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(Path("syris/gpu/cuda").glob("*")) if path.is_file()},
              "attributes": {}, "pipeline_prepare_seconds": {}, "datasets": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="syris-benchmark-") as temp:
        root = Path(temp)
        pipelines = {}
        for label in args.variants:
            print(f"Preparing {label}", flush=True)
            prepare_start = time.perf_counter()
            pipeline = (reference_pipeline(args.baseline_ref, root / label, options,
                                           uninitialized=label == "uninitialized")
                        if label in ("baseline", "uninitialized") else
                        experimental_pipeline(options, label, args.depths)
                        if label in ("hybrid", "signed", "signed_deferred", "specialized", "signed_specialized", "indexed", "scalar", "scalar_axes") or label.startswith("tile")
                        else cfg.BACKEND.pipeline)
            pipelines[label] = MeasuredPipeline(pipeline, full_grid=label in
                                                ("baseline", "uninitialized"),
                                                block_size=128 if label == "blocks128" else None,
                                                worker_blocks=4 * cp.cuda.Device().attributes["MultiProcessorCount"]
                                                if label.endswith("_pool") else None)
            names = STAGES if label == "staged" else PROJECTORS
            report["attributes"][label] = {name: pipeline.kernels[name].attributes for name in names}
            if isinstance(pipeline, DepthSpecializedPipeline):
                report["attributes"][label] = {
                    str(depth): {name: module.kernels[name].attributes for name in names}
                    for depth, module in pipeline.modules.items()}
            elapsed = time.perf_counter() - prepare_start
            report["pipeline_prepare_seconds"][label] = elapsed
            print(f"Prepared {label} in {elapsed:.1f} s (excluded from projection timing)", flush=True)
        datasets = [(path.stem, path, args.size) for path in args.mesh]
        datasets += [(f"layers-{n}", n, args.layer_size) for n in args.layers]
        for label, source, size in datasets:
            print(f"Loading {label}", flush=True)
            data = load_mesh(source) if isinstance(source, Path) else layers_mesh(source)
            # Compute oriented normals from geometry; do not let a display
            # normal filter silently reorient cavity boundaries.
            data = data.triangulate()
            faces = data.faces.reshape(-1, 4)[:, 1:]
            vertices = np.asarray(data.points, np.float64)[faces]
            normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            data.cell_data["Normals"] = normals
            mesh = Mesh.from_file(data, Trajectory([(0, 0, 0)] * q.um), center=None,
                                  unit=q.um, use_normals=True)
            print(f"Building {label}: {mesh.num_triangles} triangles, {size}x{size} detector", flush=True)
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            accel = mesh.build_accelerator()
            cp.cuda.Device().synchronize()
            item = {"dataset": label, "triangles": mesh.num_triangles, "size": size,
                    "build_seconds": time.perf_counter() - start, "views": []}
            if isinstance(source, Path):
                item.update(path=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            report["datasets"].append(item)
            xyz = np.asarray(mesh._current[:3])
            lo, hi = xyz.min(axis=1), xyz.max(axis=1)
            center, span = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
            for parallel, angle in ((True, 0), (True, 37), (False, 37)):
                camera = Camera(pixel_size=span * 1.2 * (1 if parallel else 1.75) / size * q.um,
                                shape=(size, size), trajectory=Trajectory([(0, 0, 0)] * q.um),
                                source_detector_distance=(0 if parallel else 3.5 * span) * q.um)
                camera.translate((center - [0, 0, span * (1 if parallel else 1.5)]) * q.um)
                camera.rotate_around(angle * q.deg, Y_AX, pivot=center * q.um)
                for depth in args.depths:
                    print(f"Projecting {label}: parallel={parallel}, angle={angle}, depth={depth}", flush=True)
                    view = {"parallel": parallel, "angle": angle, "depth": depth, "variants": {}}
                    item["views"].append(view)
                    images, samples = {}, {name: [] for name in pipelines}

                    def project(name):
                        accel.pipeline = pipelines[name]
                        return accel.project(camera=camera, parallel=parallel, iterations=depth,
                                             sampling_mode="staged" if name == "staged" else "local")

                    for name in pipelines:
                        project(name)
                    cp.cuda.Device().synchronize()
                    for repeat in range(args.repeats):
                        order = list(pipelines)
                        np.random.default_rng(173 + repeat).shuffle(order)
                        for name in order:
                            pipeline = pipelines[name]
                            pipeline.kernel_ms, pipeline.stage_ms = 0, {}
                            cp.cuda.Device().synchronize()
                            start = time.perf_counter()
                            images[name] = project(name)
                            cp.cuda.Device().synchronize()
                            samples[name].append({"wall_ms": (time.perf_counter() - start) * 1000,
                                                  "kernel_ms": pipeline.kernel_ms,
                                                  "stages_ms": pipeline.stage_ms.copy()})
                    reference = images[args.reference].get()
                    for name in pipelines:
                        actual = images[name].get()
                        finite = np.isfinite(actual) & np.isfinite(reference)
                        nonzero = finite & (reference != 0)
                        result = {"samples": samples[name],
                                  "median_wall_ms": float(np.median([s["wall_ms"] for s in samples[name]])),
                                  "median_kernel_ms": float(np.median([s["kernel_ms"] for s in samples[name]])),
                                  "nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
                                  "changed_finite_mask": int(np.count_nonzero(np.isfinite(actual) != np.isfinite(reference))),
                                  "changed_zero_mask": int(np.count_nonzero((actual == 0) != (reference == 0))),
                                  "changed_pixels": int(np.count_nonzero(~((actual == reference) | (np.isnan(actual) & np.isnan(reference))))),
                                  "tolerance_exceeding_pixels": int(np.count_nonzero(~np.isclose(actual, reference, atol=args.atol, rtol=args.rtol, equal_nan=True))),
                                  "max_abs_difference": float(np.max(np.abs(actual[finite] - reference[finite]), initial=0)),
                                  "max_relative_difference": float(np.max(
                                      np.abs(actual[nonzero] - reference[nonzero]) / np.abs(reference[nonzero]), initial=0))}
                        view["variants"][name] = result
                        retry_count = getattr(pipelines[name].pipeline, "last_retry_count", None)
                        if retry_count is not None:
                            result["retried_pixels"] = int(retry_count.get()[0])
                        print(json.dumps(dict(dataset=label, parallel=parallel, angle=angle, depth=depth,
                                              variant=name, **{k: v for k, v in result.items() if k != "samples"})), flush=True)
                    save()
                    # Save failures too: an invalid projection is not a speedup.
                    if any(v["nonfinite"] or v["tolerance_exceeding_pixels"]
                           for v in view["variants"].values()):
                        report["comparison_status"] = "failed"
                        save()
                        raise AssertionError("Projection is nonfinite or exceeds comparison tolerance; see saved report")
    report["comparison_status"] = "passed"
    save()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
