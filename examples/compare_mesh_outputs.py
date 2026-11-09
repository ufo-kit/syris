"""Compare TIFF outputs from matched OpenCL and CUDA mesh example runs."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def compare_directories(left, right, *, rtol=1e-5, atol=1e-12):
    """Check filenames, shapes, dtypes, finite values and every pixel; return success."""
    left, right = Path(left), Path(right)
    files_left = {p.relative_to(left) for p in left.rglob("*.tif")}
    files_right = {p.relative_to(right) for p in right.rglob("*.tif")}
    if not files_left or files_left != files_right:
        print("FAIL: TIFF file sets are empty or differ")
        print("Only in first directory:", sorted(files_left - files_right))
        print("Only in second directory:", sorted(files_right - files_left))
        return False
    passed = True
    for filename in sorted(files_left):
        a, b = imageio.imread(left / filename), imageio.imread(right / filename)
        if a.shape != b.shape or a.dtype != b.dtype:
            print(f"FAIL {filename}: shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            passed = False
            continue
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print(f"FAIL {filename}: nonfinite pixels")
            passed = False
            continue
        # Promote before subtracting, also for unsigned occupancy images.
        a, b = a.astype(np.float64), b.astype(np.float64)
        error = np.abs(a - b)
        mismatches = np.count_nonzero(error > atol + rtol * np.maximum(np.abs(a), np.abs(b)))
        ok = mismatches == 0
        passed &= ok
        print(
            f"{'PASS' if ok else 'FAIL'} {filename}: {mismatches}/{a.size} pixels outside "
            f"tolerance; max absolute difference {error.max():.9g}"
        )
    return bool(passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance in output units (metres for projections)",
    )
    args = parser.parse_args(argv)
    if any(not np.isfinite(v) or v < 0 for v in (args.rtol, args.atol)):
        parser.error("Tolerances must be finite and nonnegative")
    if not args.left.is_dir() or not args.right.is_dir():
        parser.error("Both inputs must be directories")
    return 0 if compare_directories(args.left, args.right, rtol=args.rtol, atol=args.atol) else 1


if __name__ == "__main__":
    raise SystemExit(main())
