"""Optional surface repair using libigl's CGAL Boolean operations."""

import itertools
import logging

import numpy as np

LOG = logging.getLogger(__name__)


def _cgal():
    # Keep libigl optional, including when importing the ordinary mesh loader.
    try:
        from igl.copyleft import cgal
    except ImportError as exc:
        raise ImportError(
            "Mesh repair requires libigl with CGAL support. Install the optional "
            "dependency with: uv pip install -e '.[mesh-repair]'"
        ) from exc
    return cgal


def _nondegenerate(vertices, faces):
    triangles = vertices[faces]
    return np.any(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) != 0,
        axis=1,
    )


def _require_closed_oriented(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(balance):
        raise ValueError(
            "Mesh repair requires a closed, consistently oriented surface; "
            "found open boundaries or inconsistent face winding. Orient faces "
            "out of the material (into cavities) and resolve open boundaries first."
        )


def _validate_result(cgal, vertices, faces):
    if not np.isfinite(vertices).all() or not _nondegenerate(vertices, faces).all():
        raise ValueError("The repaired surface has non-finite or degenerate triangles.")
    _require_closed_oriented(faces)
    _, _, intersections, _, _ = cgal.remesh_self_intersections(
        vertices, faces, detect_only=True, first_only=True
    )
    if len(intersections):
        raise ValueError("The repaired surface still has self-intersections.")


def _face_normals(vertices, faces):
    triangles = vertices[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _check_float32_geometry(original, converted, faces):
    if not np.isfinite(converted).all():
        raise ValueError("Repaired coordinates overflow float32.")
    # CGAL may accept coincident opposite faces even though a gap disappeared.
    if len(np.unique(converted, axis=0)) < len(np.unique(original, axis=0)):
        raise ValueError("Distinct repaired vertices merged in float32.")
    alignment = np.einsum(
        "ij,ij->i", _face_normals(original, faces), _face_normals(converted, faces)
    )
    if np.any(alignment <= 0):
        raise ValueError("Repaired triangles collapsed or flipped in float32.")


def _round_float32(
    cgal, original, faces, *, max_vertices=64, max_patch_faces=2048, max_evaluations=2048
):
    """Try neighboring Float32 coordinates without changing connectivity.

    This bounded, deterministic heuristic only addresses intersections created
    by rounding. Collapsed vertices/faces are rejected. Every coordinate stays
    at one of the two Float32 numbers bracketing its Float64 value; exactly
    representable coordinates stay fixed. Failure raises ValueError.
    """
    with np.errstate(over="ignore"):
        rounded = original.astype(np.float32).astype(np.float64)
    _check_float32_geometry(original, rounded, faces)
    pairs = cgal.remesh_self_intersections(rounded, faces, detect_only=True)[2]
    if not len(pairs):
        return rounded.astype(np.float32)

    involved = np.unique(faces[pairs])
    movable = involved[np.any(original[involved] != rounded[involved], axis=1)]
    if not len(movable) or len(movable) > max_vertices:
        raise ValueError("Float32 rounding exceeds the local vertex search limit.")

    nearest = rounded[movable].astype(np.float32)
    toward = np.where(original[movable] > rounded[movable], np.inf, -np.inf)
    with np.errstate(over="ignore"):
        neighbor = np.nextafter(nearest, toward.astype(np.float32))
    neighbor[original[movable] == rounded[movable]] = nearest[original[movable] == rounded[movable]]
    if not np.isfinite(neighbor).all():
        raise ValueError("No finite float32 bracket for repaired coordinates.")
    choices = [
        np.asarray(
            list(itertools.product(*[[a] if a == b else [a, b] for a, b in zip(start, other)])),
            dtype=np.float64,
        )
        for start, other in zip(nearest, neighbor)
    ]

    # Enclose every possible position of every affected triangle. Include all
    # triangles overlapping that box, even if they did not initially intersect.
    # There is no scene-unit tolerance or assumption about triangle size here.
    lower, upper = rounded.copy(), rounded.copy()
    lower[movable] = np.minimum(nearest, neighbor)
    upper[movable] = np.maximum(nearest, neighbor)
    incident = np.isin(faces, movable).any(axis=1)
    lo = lower[faces[incident]].min(axis=(0, 1))
    hi = upper[faces[incident]].max(axis=(0, 1))
    triangles = rounded[faces]
    nearby = np.all(triangles.max(axis=1) >= lo, axis=1) & np.all(
        triangles.min(axis=1) <= hi, axis=1
    )
    if np.count_nonzero(nearby) > max_patch_faces:
        raise ValueError("Float32 rounding exceeds the local triangle search limit.")
    ids, inverse = np.unique(faces[nearby], return_inverse=True)
    patch_faces = inverse.reshape(-1, 3).astype(np.int64)
    local_movable = np.searchsorted(ids, movable)
    reference_normals = _face_normals(original[ids], patch_faces)
    unique_count = len(np.unique(original[ids], axis=0))

    def score(vertices):
        alignment = np.einsum("ij,ij->i", reference_normals, _face_normals(vertices, patch_faces))
        invalid = np.count_nonzero(alignment <= 0) + max(
            0, unique_count - len(np.unique(vertices, axis=0))
        )
        if invalid:
            # Do not pass degenerate triangles to CGAL. This penalty exceeds
            # the number of possible intersection pairs in the patch.
            return len(patch_faces) ** 2 + invalid
        return len(cgal.remesh_self_intersections(vertices, patch_faces, detect_only=True)[2])

    LOG.info(
        "Trying bounded float32 rounding: %d intersections, %d movable "
        "vertices, %d local triangles...",
        len(pairs),
        len(movable),
        len(patch_faces),
    )
    rng = np.random.default_rng(20261105)
    best = rounded[ids].copy()
    # All initial intersection pairs lie in this patch; geometry checks above
    # have already ruled out collapsed/flipped faces and merged vertices.
    best_score = len(pairs)
    evaluations = 0
    for restart in range(4):
        if evaluations >= max_evaluations:
            break
        work = best.copy()
        current = best_score
        if restart:
            for vid, options in zip(local_movable, choices):
                work[vid] = options[rng.integers(len(options))]
            current = score(work)
            evaluations += 1
        if current < best_score:
            best, best_score = work.copy(), current
        for _ in range(6):
            improved = False
            for j in rng.permutation(len(local_movable)):
                if not best_score or evaluations >= max_evaluations:
                    break
                vid = local_movable[j]
                chosen, choice_score = work[vid].copy(), current
                for option in choices[j][rng.permutation(len(choices[j]))]:
                    if evaluations >= max_evaluations:
                        break
                    work[vid] = option
                    value = score(work)
                    evaluations += 1
                    if value < choice_score:
                        chosen, choice_score = option.copy(), value
                        if value == 0:
                            break
                work[vid] = chosen
                improved |= choice_score < current
                current = choice_score
                if current < best_score:
                    best, best_score = work.copy(), current
            if not improved or not best_score or evaluations >= max_evaluations:
                break
        if not best_score or evaluations >= max_evaluations:
            break
    if best_score:
        raise ValueError(
            "Bounded float32 rounding found no valid configuration "
            f"after {evaluations} candidate evaluations."
        )

    result = rounded.copy()
    result[ids] = best
    _check_float32_geometry(original, result, faces)
    _validate_result(cgal, result, faces)
    changed = np.count_nonzero(np.any(result != rounded, axis=1))
    displacement = np.linalg.norm(result - original, axis=1).max()
    LOG.info(
        "Float32 rounding validated: %d vertices adjusted, %d evaluations, "
        "maximum displacement %.6g in mesh units.",
        changed,
        evaluations,
        displacement,
    )
    return result.astype(np.float32)


def repair_mesh(vertices, faces, dtype=np.float64, *, float32_rounding=False):
    """Return the boundary of the positive-winding material region.

    ``vertices`` is an (N, 3) array; ``faces`` is an (M, 3) integer array.
    All shells of ONE material must be supplied together, with face winding
    pointing out of material, including into cavities. Self-intersections and
    overlaps are resolved by a Boolean self-union, retaining regions with
    winding number > 0. Correctly oriented cavities and disconnected material
    islands are preserved; separate materials must be processed separately.

    Exactly coincident vertices are welded and zero-area faces are removed.
    Open boundaries and inconsistent orientation are rejected, not inferred.
    libigl computes in Float64; the result is checked for degeneracy, oriented
    closure and intersections again after conversion to ``dtype``. Loss of
    validity raises ValueError. This check covers the returned coordinates,
    not subsequent transforms or unit conversions.

    ``float32_rounding=True`` enables experimental bounded rounding when
    ``dtype`` is Float32. It tries the two Float32 numbers bracketing each
    affected coordinate, preserving connectivity and checking the whole result.
    Collapsed vertices/faces and exhausted searches still raise ValueError.

    Returns new ``(vertices, faces)`` arrays. Input arrays are not modified.
    This operation changes connectivity and does not transfer mesh attributes.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Mesh repair supports float32 and float64 coordinates.")
    if float32_rounding and dtype != np.dtype(np.float32):
        raise ValueError("float32_rounding requires dtype=np.float32.")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Expected a nonempty, finite (N, 3) vertex array.")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Expected a nonempty (M, 3) array of valid integer vertex indices.")

    cgal = _cgal()
    original_count = len(faces)
    # Do not use a distance tolerance: nearby vertices may delimit a thin wall.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = np.asarray(inverse[faces], dtype=np.int64)
    keep = _nondegenerate(vertices, faces)
    faces = faces[keep]
    if not len(faces):
        raise ValueError("Mesh repair found no nondegenerate triangles.")
    _require_closed_oriented(faces)

    LOG.info(
        "Repairing %d triangles with libigl (%d zero-area faces removed)...",
        original_count,
        original_count - len(faces),
    )
    try:
        vertices, faces, _ = cgal.mesh_boolean(
            vertices,
            faces,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            "union",
        )
    except RuntimeError as exc:
        raise ValueError("libigl could not resolve this mesh's material boundary.") from exc
    if not len(faces):
        raise ValueError(
            "Mesh repair produced no positive-winding material. Check face "
            "orientation; exterior surfaces must point outward."
        )

    _validate_result(cgal, vertices, faces)
    with np.errstate(over="ignore"):
        converted = vertices.astype(dtype)
    if dtype == np.dtype(np.float32):
        try:
            if float32_rounding:
                converted = _round_float32(cgal, vertices, faces)
            else:
                _check_float32_geometry(vertices, converted.astype(np.float64), faces)
                _validate_result(cgal, converted.astype(np.float64), faces)
        except ValueError as exc:
            rounding_hint = (
                "Try experimental float32_rounding=True (--float32-rounding) "
                "to search for a valid Float32 conversion. "
                if not float32_rounding
                else ""
            )
            raise ValueError(
                f"The repaired mesh is invalid after conversion to float32: {exc} "
                f"{rounding_hint}"
                "Use syris.init(double_precision=True), or --double-precision "
                "in examples.mesh_cuda."
            ) from exc
    LOG.info("Mesh repair complete: %d triangles, %s coordinates.", len(faces), dtype)
    return converted, faces
