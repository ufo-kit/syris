import argparse
import json
from pathlib import Path

import numpy as np

from . import probe as p
from . import validate as v


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce and isolate the two adverse rounding variants"
    )
    p.add_path_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    p.configure_paths(args)
    import cupy as cp

    with np.load(p.BENCH / "data/prepared/bunny/float32-baseline.npz") as data:
        points, faces = data["points"], data["faces"]
    fov = json.loads((p.BENCH / "data/baseline/bunny.json").read_text())["fov_um"]
    modes = {}
    for mode in ["float32", "mixed", "float64"]:
        h = p.ProjectionHarness(double=mode == "float64", mixed_precision=mode == "mixed")
        h.mixed = mode == "mixed"
        modes[mode] = h
    originals = {name: h.pipeline for name, h in modes.items()}
    compiled = {}
    records = []
    for angle, centered, scaling, rays in [
        (45, True, "power2", "centered"),
        (90, False, "power2", "current"),
    ]:
        ref, _ = v.image(
            modes["float64"],
            originals["float64"],
            points,
            faces,
            fov,
            2048,
            angle,
            False,
            "current",
            reps=1,
        )
        baseline, _ = v.image(
            modes["float32"],
            originals["float32"],
            points,
            faces,
            fov,
            2048,
            angle,
            False,
            "current",
            reps=1,
        )
        pixel = None
        for mode in ["float32", "mixed"]:
            h = modes[mode]
            key = (mode, rays)
            if key not in compiled:
                compiled[key] = p.pipeline_for(h, rays)
            capture = p.CapturePipeline(compiled[key])
            frame, times = v.image(
                h, capture, points, faces, fov, 2048, angle, centered, scaling, reps=1
            )
            metrics, _ = v.compare(frame, ref)
            if pixel is None:
                pixel = tuple(
                    int(i)
                    for i in np.unravel_index(
                        np.nanargmax(np.abs(frame.astype(float) - ref)), frame.shape
                    )
                )
            a = capture.last_args
            origin = cp.empty((1, 4), h.dtype)
            compiled[key].kernels["capture_rounding_rays"](
                (1,), (32,), (np.int32(1), cp.asarray([pixel], np.uint32), *a[4:10], origin)
            )
            origin = origin.get()[0]
            vertices = a[3].view(h.dtype).reshape(-1, 4).get()[:, :3]
            encoded = (
                p.decimal_reference(vertices, np.arange(len(vertices)).reshape(-1, 3), origin[:2])
                * 128
            )
            rec = dict(
                angle=angle,
                centered=centered,
                ray_mode=rays,
                mode=mode,
                pixel=pixel,
                original_um=float(baseline[pixel]) * 1e6,
                reference_um=float(ref[pixel]) * 1e6,
                variant_um=float(frame[pixel]) * 1e6,
                encoded_reference_um=encoded,
                metrics=metrics,
            )
            # Direct triangle tests after reproducing the production z-origin rebasing.
            lower = a[13].view(h.dtype).reshape(-1, 4).get()[len(faces), :3]
            upper = a[14].view(h.dtype).reshape(-1, 4).get()[len(faces), :3]
            origin[2] = h.dtype(
                float(lower[2])
                - 32
                * np.finfo(h.dtype).eps
                * float(np.max(upper.astype(float) - lower.astype(float)))
            )
            h.pipeline = originals[mode]
            flags, dist = h.triangles(
                np.tile(origin[:3], (len(faces), 1)),
                np.tile([0, 0, 1], (len(faces), 1)),
                vertices.reshape(-1, 3, 3),
            )
            inds = np.flatnonzero(flags[:, 0])
            rec["triangle_hits"] = [
                dict(
                    face=int(i),
                    distance=float(dist[i]),
                    world_z_um=(float(dist[i]) + float(origin[2])) * 128,
                )
                for i in inds
            ]
            records.append(rec)
            print(json.dumps(rec), flush=True)
    output = args.output or p.ROOT / "regressions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(provenance=p.provenance(), results=records), indent=2))


if __name__ == "__main__":
    main()
