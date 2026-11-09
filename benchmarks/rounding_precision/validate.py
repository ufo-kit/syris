"""Full-frame comparison plus independent CPU plane references on held-out rays."""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import quantities as q

from syris.geometry import Y_AX

from . import probe as p


def plane_references(points, faces, xy):
    tri = np.asarray(points, dtype=np.float64)[faces]
    lower = np.nextafter(tri[:, :, :2].min(axis=1), -np.inf)
    upper = np.nextafter(tri[:, :, :2].max(axis=1), np.inf)
    results = []
    ld = np.longdouble
    for x, y in xy:
        candidate = (
            (lower[:, 0] <= x) & (upper[:, 0] >= x) & (lower[:, 1] <= y) & (upper[:, 1] >= y)
        )
        t = tri[candidate].astype(ld)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        e, f = b - a, c - a
        normal = np.cross(e, f)
        signs = np.stack(
            [
                (b[:, 0] - a[:, 0]) * (ld(y) - a[:, 1]) - (b[:, 1] - a[:, 1]) * (ld(x) - a[:, 0])
                for a, b in [(a, b), (b, c), (c, a)]
            ],
            axis=1,
        )
        inside = np.all(signs >= 0, axis=1) | np.all(signs <= 0, axis=1)
        if np.any(inside & np.any(signs == 0, axis=1)):
            results.append(np.nan)
            continue
        valid = inside & (normal[:, 2] != 0)
        a, normal = a[valid], normal[valid]
        z = (
            a[:, 2]
            - (normal[:, 0] * (ld(x) - a[:, 0]) + normal[:, 1] * (ld(y) - a[:, 1])) / normal[:, 2]
        )
        z.sort()
        results.append(float(np.sum(z[1::2] - z[::2])) if len(z) % 2 == 0 else np.nan)
    return np.array(results) * 1e-6


def image(h, compiled, points, faces, fov, n, angle, centered, scaling, reps=3):
    import cupy as cp

    h.pipeline = compiled
    with h.configured():
        mesh = p.make_mesh(points, faces)
        scan = p.ScanProjector(
            mesh,
            fov,
            n,
            SimpleNamespace(supersampling=1, cuda_adaptive_depth=0),
            "cuda",
            "rotating",
        )
        scan.pose(angle)
        camera = scan.projector.camera
        if centered:
            mesh.clear_transformation()
            mesh.rotate(angle * q.deg, Y_AX)
            distance = float(camera.position.rescale(q.um).magnitude[2])
            camera.clear_transformation()
            camera.translate(np.array([0.0, 0.0, distance]) * q.um)
        if scaling == "power2":
            mesh._furthest_point = 2.0 ** math.ceil(math.log2(mesh._furthest_point))
        elif scaling == "none":
            mesh._furthest_point = 1.0
        accel = p.BvhCupyAccelerator(mesh)
        accel.build()
        times = []
        for rep in range(reps + 1):
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            frame = accel.project(camera=camera, parallel=True, iterations=0).get()
            frame *= 1e-6
            cp.cuda.get_current_stream().synchronize()
            if rep:
                times.append((time.perf_counter() - start) * 1e3)
        return frame, times


def compare(value, reference):
    valid = np.isfinite(value) & np.isfinite(reference)
    occupied = valid & (reference > 1e-10)
    error = np.abs(value.astype(np.float64) - reference)
    exceeds = valid & (error > 1e-9 + 1e-4 * np.abs(reference))
    k = min(16, error.size)
    index = np.argpartition(np.where(valid, error, -1).ravel(), -k)[-k:]
    worst = np.column_stack(np.unravel_index(index, error.shape))
    return dict(
        nonfinite=int(np.sum(~np.isfinite(value))),
        negative=int(np.sum(value < 0)),
        exceeding_pixels=int(exceeds.sum()),
        foreground_pixels=int(occupied.sum()),
        false_zero=int(np.sum(occupied & (value == 0))),
        extra_positive=int(np.sum(valid & (reference == 0) & (value > 1e-10))),
        max_error_nm=float(np.max(error[valid]) * 1e9),
        foreground_p99_error_nm=float(np.percentile(error[occupied], 99) * 1e9),
        foreground_rmse_nm=float(np.sqrt(np.mean(error[occupied] ** 2)) * 1e9),
    ), worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    p.add_path_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p.configure_paths(args)
    gp = {}
    for mode in ["float32", "mixed", "float64"]:
        h = p.ProjectionHarness(double=mode == "float64", mixed_precision=mode == "mixed")
        h.mixed = mode == "mixed"
        gp[mode] = h
    compiled = {
        (mode, rays): p.pipeline_for(gp[mode], rays)
        for mode, rays in [
            ("float32", "current"),
            ("float32", "centered"),
            ("mixed", "current"),
            ("float64", "current"),
        ]
    }
    variants = [
        ("current", "float32", False, "current", "current"),
        ("power2", "float32", False, "power2", "current"),
        ("centered_power2", "float32", True, "power2", "current"),
        ("centered_power2_rays", "float32", True, "power2", "centered"),
        ("mixed", "mixed", False, "current", "current"),
        ("centered_power2_mixed", "mixed", True, "power2", "current"),
    ]
    results = []
    document = dict(
        provenance=p.provenance(),
        scope="Same prepared Float32 geometry for every variant, including Float64 comparison. Parallel single-sample images. Timing is warmed accelerator.project plus D2H and metre conversion, excluding BVH build and pose.",
        tolerance={"absolute_m": 1e-9, "relative": 1e-4},
        seed=20261109,
        results=results,
    )
    path = args.output or p.ROOT / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(document["seed"])
    for meshname, n in [("bunny", 2048), ("armadillo", 512)]:
        with np.load(p.BENCH / f"data/prepared/{meshname}/float32-baseline.npz") as data:
            points, faces = data["points"], data["faces"]
        fov = json.loads((p.BENCH / f"data/baseline/{meshname}.json").read_text())["fov_um"]
        for angle in [0, 17, 45, 90, 135]:
            print("Reference", meshname, n, angle, flush=True)
            reference, _ = image(
                gp["float64"],
                compiled["float64", "current"],
                points,
                faces,
                fov,
                n,
                angle,
                False,
                "current",
                reps=1,
            )
            foreground = np.flatnonzero(reference.ravel() > 1e-10)
            picked = rng.choice(foreground, min(256, len(foreground)), replace=False)
            selected = np.column_stack(np.unravel_index(picked, reference.shape))
            selected = np.concatenate([selected, rng.integers(0, n, (64, 2))])
            if meshname == "bunny" and angle == 0:
                selected = np.concatenate(
                    [
                        selected,
                        np.array(
                            [
                                (r + dr, c + dc)
                                for r, c in p.PIXELS
                                for dr in range(-2, 3)
                                for dc in range(-2, 3)
                            ]
                        ),
                    ]
                )
            theta = np.deg2rad(angle)
            rot = np.array(
                [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
            )
            rotated = points.astype(np.float64) @ rot.T
            xy = (selected[:, ::-1].astype(float) + 0.5) * fov / n - fov / 2
            independent = plane_references(rotated, faces, xy)
            established = np.isfinite(independent)
            discrepancies = (
                np.abs(reference[tuple(selected.T)][established] - independent[established]) * 1e9
            )
            check = dict(
                selected_rays=len(selected),
                established_rays=int(established.sum()),
                max_float64_reference_error_nm=float(np.max(discrepancies)),
            )
            if check["max_float64_reference_error_nm"] > 1e-3:
                raise RuntimeError(("Float64 reference validation failed", meshname, angle, check))
            print("Independent CPU check", check, flush=True)
            order = list(range(len(variants)))
            rng.shuffle(order)
            for v in order:
                name, mode, centered, scaling, rays = variants[v]
                print("Variant", meshname, angle, name, flush=True)
                frame, times = image(
                    gp[mode], compiled[mode, rays], points, faces, fov, n, angle, centered, scaling
                )
                metrics, worst = compare(frame, reference)
                worstxy = (worst[:, ::-1].astype(float) + 0.5) * fov / n - fov / 2
                worstref = plane_references(rotated, faces, worstxy)
                okay = np.isfinite(worstref)
                if (
                    np.any(okay)
                    and np.max(np.abs(reference[tuple(worst.T)][okay] - worstref[okay])) > 1e-12
                ):
                    raise RuntimeError(
                        ("Worst-pixel Float64 reference check failed", meshname, angle, name)
                    )
                subseterror = np.abs(
                    frame[tuple(selected.T)][established] - independent[established]
                )
                record = dict(
                    worst_pixels=worst.tolist(),
                    mesh=meshname,
                    resolution=n,
                    angle_deg=angle,
                    variant=name,
                    **metrics,
                    projection_host_ms=times,
                    independent_reference_check=check,
                    independent_exceeding_pixels=int(
                        np.sum(subseterror > 1e-9 + 1e-4 * np.abs(independent[established]))
                    ),
                    worst_selected_established=int(okay.sum()),
                    worst_reference_discrepancy_nm=float(
                        np.max(np.abs(reference[tuple(worst.T)][okay] - worstref[okay])) * 1e9
                    )
                    if np.any(okay)
                    else None,
                )
                results.append(record)
                path.write_text(json.dumps(document, indent=2))
                print(json.dumps(record), flush=True)
    print("Wrote", path, flush=True)


if __name__ == "__main__":
    main()
