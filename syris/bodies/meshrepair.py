"""Optional surface repair using libigl's CGAL Boolean operations."""

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
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]) != 0,
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


def repair_mesh(vertices, faces, dtype=np.float64):
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

    Returns new ``(vertices, faces)`` arrays. Input arrays are not modified.
    This operation changes connectivity and does not transfer mesh attributes.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Mesh repair supports float32 and float64 coordinates.")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError("Expected a nonempty, finite (N, 3) vertex array.")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
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

    LOG.info("Repairing %d triangles with libigl (%d zero-area faces removed)...",
             original_count, original_count - len(faces))
    try:
        vertices, faces, _ = cgal.mesh_boolean(
            vertices, faces,
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
    converted = vertices.astype(dtype)
    if dtype == np.dtype(np.float32):
        try:
            # Intersection predicates may accept coincident opposite faces.
            # Newly merged vertices can nevertheless erase a gap or cavity.
            if len(np.unique(converted, axis=0)) < len(np.unique(vertices, axis=0)):
                raise ValueError("Distinct repaired vertices merged in float32.")
            _validate_result(cgal, converted.astype(np.float64), faces)
        except ValueError as exc:
            raise ValueError(
                "The repaired mesh is invalid after conversion to float32. "
                "Use syris.init(double_precision=True), or --double-precision "
                "in examples.mesh_cuda."
            ) from exc
    LOG.info("Mesh repair complete: %d triangles, %s coordinates.", len(faces), dtype)
    return converted, faces
