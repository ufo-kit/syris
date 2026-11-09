"""Test Float64 hit storage with Float32 geometry and mixed triangle arithmetic."""

import argparse
import json
import unittest
from pathlib import Path

import numpy as np

from syris.tests.test_cuda_projection import ProjectionCases

from . import probe as p
from . import validate as v


def compile_wide(harness):
    directory = p.ROOT / "sources" / "wide_hits"
    directory.mkdir(parents=True, exist_ok=True)
    for original in (p.REPO / "syris/gpu/cuda").iterdir():
        if original.suffix not in (".cu", ".cuh"):
            continue
        code = original.read_text()
        if original.name == "Commons.cuh":
            code = code.replace("    FP_T t;", "    double t;")
            code = code.replace("HitRecord(FP_T t_val", "HitRecord(double t_val")
        elif original.name in ("WatertightRay.cuh", "WatertightRay.cu"):
            code = code.replace("FP_T &t, unsigned", "double &t, unsigned")
            code = code.replace("t = FP_T(distance);", "t = distance;")
        elif original.name == "source.cu":
            for old, new in [
                ("FP_T entry = 0, last = 0;", "double entry = 0, last = 0;"),
                ("event(FP_T t, int orientation)", "event(double t, int orientation)"),
                ("FP_T t;", "double t;"),
                ("FP_T previous = NEG_INFINITY;", "double previous = NEG_INFINITY;"),
                ("FP_T t = hits.values[i].t;", "double t = hits.values[i].t;"),
                ("FP_T first = POS_INFINITY;", "double first = POS_INFINITY;"),
            ]:
                assert old in code, old
                code = code.replace(old, new)
            start = code.index('extern "C" __global__ void project_parallel_kernel(')
            end = code.index('extern "C" __global__ void project_conebeam_kernel(')
            code = code[:start] + code[start:end].replace(p.OLD_RAY, p.NEW_RAY) + code[end:]
        (directory / original.name).write_text(code)
    tests = directory / "projection_probe.cu"
    tests.write_text(
        (p.REPO / "syris/tests/cuda/projection_probe.cu")
        .read_text()
        .replace("FP_T t = -1;", "double t = -1;")
    )
    pipeline = p.CudaPipeline([str(directory)], ["-DSYRIS_MIXED_PRECISION"])
    pipeline.readModuleFromFiles(
        "wide_hits",
        [directory / name for name in ("WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu")]
        + [tests],
    )
    for name in harness.pipeline.kernels:
        pipeline.getKernelFromModule("wide_hits", name)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    p.add_path_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p.configure_paths(args)
    h = p.ProjectionHarness(mixed_precision=True)
    h.mixed = True
    reference_h = p.ProjectionHarness(double=True)
    reference_h.mixed = False
    print("Compile Float64 hit storage prototype", flush=True)
    pipeline = compile_wide(h)

    class WideHitsRegression(ProjectionCases, unittest.TestCase):
        double = False
        mixed_precision = True

        @classmethod
        def setUpClass(cls):
            cls.gpu = h
            cls.gpu.pipeline = pipeline

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WideHitsRegression)
    checked = unittest.TextTestRunner(verbosity=1).run(suite)
    if not checked.wasSuccessful():
        raise RuntimeError("Prototype failed existing CUDA projection regressions")
    document = dict(
        provenance=p.provenance(),
        variant="centered_power2_rays_mixed_wide_hits",
        test_count=checked.testsRun,
        results=[],
    )
    path = args.output or p.ROOT / "wide_hits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261109)
    for meshname, n in [("bunny", 2048), ("armadillo", 512)]:
        with np.load(p.BENCH / f"data/prepared/{meshname}/float32-baseline.npz") as data:
            points, faces = data["points"], data["faces"]
        fov = json.loads((p.BENCH / f"data/baseline/{meshname}.json").read_text())["fov_um"]
        for angle in [0, 17, 45, 90, 135]:
            print("Validate wide hits", meshname, angle, flush=True)
            reference, _ = v.image(
                reference_h,
                reference_h.pipeline,
                points,
                faces,
                fov,
                n,
                angle,
                False,
                "current",
                reps=1,
            )
            frame, times = v.image(h, pipeline, points, faces, fov, n, angle, True, "power2")
            metrics, worst = v.compare(frame, reference)
            foreground = np.flatnonzero(reference.ravel() > 1e-10)
            indices = rng.choice(foreground, min(256, len(foreground)), replace=False)
            selected = np.concatenate(
                [np.column_stack(np.unravel_index(indices, reference.shape)), worst]
            )
            theta = np.deg2rad(angle)
            rotation = np.array(
                [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
            )
            xy = (selected[:, ::-1].astype(float) + 0.5) * fov / n - fov / 2
            independent = v.plane_references(points.astype(float) @ rotation.T, faces, xy)
            valid = np.isfinite(independent)
            ref_error = np.abs(reference[tuple(selected.T)][valid] - independent[valid])
            if np.max(ref_error) > 1e-12:
                raise RuntimeError("Independent reference discrepancy exceeds 0.001 nm")
            record = dict(
                mesh=meshname,
                resolution=n,
                angle_deg=angle,
                **metrics,
                projection_host_ms=times,
                independent_reference_rays=int(valid.sum()),
                independent_reference_error_nm=float(np.max(ref_error) * 1e9),
                worst_pixels=worst.tolist(),
            )
            if meshname == "bunny" and angle == 45:
                record["previous_regression_pixel_um"] = float(frame[1291, 652]) * 1e6
                record["previous_regression_reference_um"] = float(reference[1291, 652]) * 1e6
            document["results"].append(record)
            path.write_text(json.dumps(document, indent=2))
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
