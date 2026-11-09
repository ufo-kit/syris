"""Compare public precision modes on the prepared Bunny and Armadillo meshes."""

import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--write-references", action="store_true")
    parser.add_argument("--legacy-root", type=Path, help="Optional pre-change source checkout")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.legacy_root:
        sys.path.insert(0, str(args.legacy_root.resolve()))

    import cupy as cp

    import syris
    from syris import config as cfg

    from .probe import PIXELS, ScanProjector, make_mesh
    from .validate import compare, plane_references

    root = Path(syris.__file__).resolve().parent.parent
    files = list((root / "syris/gpu/cuda").glob("*.cu*"))
    files += [
        root / name
        for name in (
            "syris/__init__.py",
            "syris/config.py",
            "syris/bodies/accelerators.py",
            "syris/bodies/mesh.py",
            "syris/devices/cameras.py",
            "examples/mesh_projection.py",
        )
    ]
    inputs = [
        args.benchmark_root / f"data/prepared/{mesh}/float32-baseline.npz"
        for mesh in ("bunny", "armadillo")
    ]
    inputs += [
        args.benchmark_root / f"data/baseline/{mesh}.json" for mesh in ("bunny", "armadillo")
    ]
    document = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv,
        "implementation_root": str(root),
        "legacy": bool(args.legacy_root),
        "source_sha256": {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "input_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "timing": "warmed wall time; projection includes D2H and metre conversion; no TIFF I/O",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.write_references:
        args.references.mkdir(parents=True, exist_ok=True)

    for mode in ("float64", "float32", "mixed"):
        options = (
            dict(double_precision=mode == "float64", mixed_precision=mode == "mixed")
            if args.legacy_root
            else dict(precision=mode)
        )
        syris.init(compute_backend="cuda", loglevel=logging.ERROR, **options)
        if cfg.BACKEND.name != cfg.BACKEND.CUDA:
            raise RuntimeError("The requested CUDA backend is unavailable")
        for meshname, n in (("bunny", 2048), ("armadillo", 512)):
            with np.load(
                args.benchmark_root / f"data/prepared/{meshname}/float32-baseline.npz"
            ) as data:
                points, faces = data["points"], data["faces"]
            fov = json.loads((args.benchmark_root / f"data/baseline/{meshname}.json").read_text())[
                "fov_um"
            ]
            for angle in (0, 17, 45, 90, 135):
                mesh = make_mesh(points, faces)
                scan = ScanProjector(
                    mesh,
                    fov,
                    n,
                    SimpleNamespace(supersampling=1, cuda_adaptive_depth=0),
                    "cuda",
                    "rotating",
                )
                scan.pose(angle)
                accel = mesh.build_accelerator()
                builds = []
                for _ in range(3):
                    cp.cuda.get_current_stream().synchronize()
                    start = time.perf_counter()
                    accel.build()
                    cp.cuda.get_current_stream().synchronize()
                    builds.append((time.perf_counter() - start) * 1e3)
                times = []
                for repeat in range(args.repeats + 1):
                    cp.cuda.get_current_stream().synchronize()
                    start = time.perf_counter()
                    frame = accel.project(camera=scan.projector.camera, parallel=True).get()
                    frame *= 1e-6
                    cp.cuda.get_current_stream().synchronize()
                    if repeat:
                        times.append((time.perf_counter() - start) * 1e3)
                path = args.references / f"{meshname}-{angle}.npy"
                if args.write_references and mode == "float64":
                    np.save(path, frame)
                reference = np.load(path)
                metrics, worst = compare(frame, reference)
                rng = np.random.default_rng(20261109)
                occupied = np.flatnonzero(reference.ravel() > 1e-10)
                selected = rng.choice(occupied, min(32, len(occupied)), replace=False)
                selected = np.concatenate(
                    [np.column_stack(np.unravel_index(selected, frame.shape)), worst]
                )
                theta = np.deg2rad(angle)
                rotation = np.array(
                    [
                        [np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)],
                    ]
                )
                xy = (selected[:, ::-1].astype(float) + 0.5) * fov / n - fov / 2
                independent = plane_references(points.astype(float) @ rotation.T, faces, xy)
                valid = np.isfinite(independent)
                discrepancy = np.max(
                    np.abs(reference[tuple(selected.T)][valid] - independent[valid])
                )
                if discrepancy > 1e-12:
                    raise RuntimeError("Independent reference discrepancy exceeds 0.001 nm")
                record = dict(
                    mode=mode,
                    mesh=meshname,
                    angle_deg=angle,
                    resolution=n,
                    **metrics,
                    build_ms=builds,
                    projection_host_ms=times,
                    independent_reference_rays=int(valid.sum()),
                    independent_reference_max_error_nm=float(discrepancy * 1e9),
                )
                if meshname == "bunny":
                    coordinates = {0: PIXELS, 45: [(1291, 652)], 90: [(1138, 1220)]}.get(angle, [])
                    record["audited_pixels"] = [
                        dict(
                            row=int(row),
                            col=int(col),
                            thickness_um=float(frame[row, col]) * 1e6,
                            reference_um=float(reference[row, col]) * 1e6,
                        )
                        for row, col in coordinates
                    ]
                document["results"].append(record)
                args.output.write_text(json.dumps(document, indent=2))
                print(
                    f"{mode} {meshname} {angle}: {metrics['exceeding_pixels']} over tolerance; "
                    f"max {metrics['max_error_nm']:.6f} nm; "
                    f"{np.median(times):.3f} ms incl. transfer",
                    flush=True,
                )


if __name__ == "__main__":
    main()
