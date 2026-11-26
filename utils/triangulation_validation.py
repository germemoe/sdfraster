"""Fail-closed CPU validation of Delaunay input and tetrahedra.

This checks finite nondegenerate positive-volume tetrahedra, not Delaunayness
or manifoldness. No point is deduplicated, perturbed or moved.
"""
from __future__ import annotations
import hashlib
import numbers
import numpy as np

# Optional exact duplicate-coordinate counter (native/exact_triplet_counts.cpp, built by
# scripts/build_native.py). It returns the same counts as the NumPy sort; without the
# library the NumPy path is used.
from .exact_triplet_counts import load_exact_triplet_counts as _load_exact_triplet_counts
_exact_triplet_counts = _load_exact_triplet_counts()


class TriangulationValidationError(ValueError):
    def __init__(self, reason, message, metadata=None):
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.metadata = dict(metadata or {})
        self.metadata.update(validation_status="rejected", failure_reason=reason)


def _reject(reason, message, metadata):
    raise TriangulationValidationError(reason, message, metadata)


def _prepare_points(points):
    meta = {"validation_status": "pending"}
    try:
        original = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise TriangulationValidationError("point_array", str(exc), meta) from exc
    meta.update(input_dtype=str(original.dtype), input_shape=list(original.shape))
    if original.ndim != 2 or original.shape[1] != 3:
        _reject("point_shape", "points must have shape (N, 3)", meta)
    if original.dtype.kind not in "fiu":
        _reject("point_dtype", "coordinates must be real numeric values", meta)
    if original.dtype.kind == "f" and original.dtype.itemsize > 8:
        _reject("point_dtype", "extended precision coordinates are not converted silently", meta)
    if original.dtype.kind in "iu" and original.size:
        if np.any(original > 2**53) or (original.dtype.kind == "i" and np.any(original < -(2**53))):
            _reject("point_dtype", "integer coordinates exceed exact float64 range", meta)
    array = np.ascontiguousarray(original, dtype=np.float64)
    n = len(array)
    meta.update(point_count=n, coordinate_dtype="float64",
                coordinate_sha256=hashlib.sha256(array).hexdigest())
    if n < 4:
        _reject("too_few_points", "at least four points are required", meta)
    if n > np.iinfo(np.int32).max:
        _reject("point_count", "point indices cannot fit int32", meta)
    finite = np.isfinite(array)
    if not finite.all():
        bad = np.argwhere(~finite)
        meta.update(nonfinite_coordinate_count=len(bad), nonfinite_coordinate_examples=bad[:16].tolist())
        _reject("nonfinite_points", "coordinates contain NaN or infinity", meta)
    # Exact duplicate diagnosis is independent of numerical affine rank.
    unique_count, duplicate_group_count = _exact_triplet_counts(points, array)
    meta.update(unique_point_count=unique_count, duplicate_point_count=n - unique_count,
                duplicate_coordinate_group_count=duplicate_group_count,
                bbox_min=array.min(axis=0).tolist(), bbox_max=array.max(axis=0).tolist())
    if unique_count < 4:
        _reject("too_few_unique_points", "fewer than four distinct coordinates", meta)
    # Normalize before subtraction, preventing overflow in the rank diagnostic.
    scale = float(np.max(np.abs(array)))
    normalized = array / scale if scale else array.copy()
    centered = normalized - normalized[0]
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise TriangulationValidationError("point_rank", str(exc), meta) from exc
    tolerance = float(max(centered.shape) * np.finfo(np.float64).eps * singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    meta.update(affine_rank=rank, rank_coordinate_scale=scale,
                rank_singular_values=singular.tolist(), rank_tolerance=tolerance)
    if rank < 3:
        _reject("rank_deficient_points", f"numerical affine rank is {rank}, expected 3", meta)
    return array, meta


def diagnose_points(points):
    """Diagnose usable 3D input; exact duplicates are recorded, not removed."""
    _, meta = _prepare_points(points)
    meta["validation_status"] = "input_validated"
    return meta


def validate_tetrahedra(points, raw, backend="native", chunk_size=65536):
    """Return new contiguous int32 indices and JSON-safe diagnostics.

    Native sentinel N rows are removed, but invalid values/repeated IDs even in
    those rows reject the whole candidate. SciPy accepts only indices below N.
    Orientation matches topo_utils.tet_volumes in original float64 coordinates.
    """
    if backend not in ("native", "scipy"):
        raise ValueError("backend must be 'native' or 'scipy'")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    coordinates, meta = _prepare_points(points)
    meta.update(backend=backend, chunk_size=int(chunk_size))
    try:
        tetrahedra = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise TriangulationValidationError("tetrahedron_array", str(exc), meta) from exc
    meta.update(raw_dtype=str(tetrahedra.dtype), raw_shape=list(tetrahedra.shape))
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        _reject("tetrahedron_shape", "indices must have shape (M, 4)", meta)
    if tetrahedra.dtype.kind not in "fiu":
        _reject("tetrahedron_dtype", "indices must be real numeric integers", meta)
    if len(tetrahedra) == 0:
        _reject("empty_tetrahedra", "candidate contains no tetrahedra", meta)
    n = len(coordinates)
    max_allowed = n if backend == "native" else n - 1
    output = np.empty((len(tetrahedra), 4), dtype=np.int32)
    used = np.zeros(n, dtype=bool)
    accepted = infinity_rows = flips = 0
    minimum, maximum = float("inf"), 0.0
    for start in range(0, len(tetrahedra), int(chunk_size)):
        block = tetrahedra[start:start + int(chunk_size)]
        if block.dtype.kind == "f":
            bad = ~np.isfinite(block)
            if bad.any():
                meta["bad_raw_row"] = int(start + np.argwhere(bad)[0, 0])
                _reject("nonfinite_indices", "index contains NaN or infinity", meta)
            bad = block != np.floor(block)
            if bad.any():
                meta["bad_raw_row"] = int(start + np.argwhere(bad)[0, 0])
                _reject("noninteger_indices", "raw index is not an integer", meta)
        bad = (block < 0) | (block > max_allowed)
        if bad.any():
            row, col = np.argwhere(bad)[0]
            meta.update(bad_raw_row=int(start + row), bad_raw_column=int(col), bad_raw_index=int(block[row, col]))
            _reject("index_out_of_range", f"indices must lie in [0, {max_allowed}]", meta)
        indices = block.astype(np.int64, copy=True)
        repeated = (
            (indices[:, 0] == indices[:, 1]) | (indices[:, 0] == indices[:, 2]) |
            (indices[:, 0] == indices[:, 3]) | (indices[:, 1] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 3]) | (indices[:, 2] == indices[:, 3])
        )
        if repeated.any():
            meta["bad_raw_row"] = int(start + np.flatnonzero(repeated)[0])
            _reject("repeated_tetrahedron_index", "tetrahedron has repeated vertex IDs", meta)
        infinity = np.any(indices == n, axis=1) if backend == "native" else np.zeros(len(indices), bool)
        block_infinity = int(infinity.sum())
        infinity_rows += block_infinity
        if block_infinity:
            indices = indices[~infinity]
        if not len(indices):
            continue
        edges = coordinates[indices[:, 1:]]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            edges -= coordinates[indices[:, :1]]
            determinants = np.linalg.det(edges)
            volumes = determinants / 6.0
        finite = np.isfinite(edges).all(axis=(1, 2)) & np.isfinite(determinants) & np.isfinite(volumes)
        if not finite.all():
            kept_rows = np.flatnonzero(~infinity)
            meta["bad_raw_row"] = int(start + kept_rows[np.flatnonzero(~finite)[0]])
            _reject("nonfinite_volume", "tetrahedron edges or signed volume are not finite", meta)
        zero = volumes == 0
        if zero.any():
            kept_rows = np.flatnonzero(~infinity)
            meta["bad_raw_row"] = int(start + kept_rows[np.flatnonzero(zero)[0]])
            _reject("zero_volume", "tetrahedron has zero float64 signed volume", meta)
        negative = volumes < 0
        if negative.any():
            first = indices[negative, 0].copy()
            indices[negative, 0] = indices[negative, 1]
            indices[negative, 1] = first
            flips += int(negative.sum())
        absolute = np.abs(volumes)
        minimum, maximum = min(minimum, float(absolute.min())), max(maximum, float(absolute.max()))
        output[accepted:accepted + len(indices)] = indices
        accepted += len(indices)
        used[indices.ravel()] = True
    if accepted == 0:
        meta["infinity_tetrahedron_count"] = infinity_rows
        _reject("empty_finite_tetrahedra", "no finite tetrahedra remain after infinity filtering", meta)
    omitted = np.flatnonzero(~used)
    result = np.ascontiguousarray(output[:accepted])
    meta.update(validation_status="validated", raw_tetrahedron_count=len(tetrahedra),
                tetrahedron_count=accepted, infinity_tetrahedron_count=infinity_rows,
                orientation_flip_count=flips, positive_volume_min=minimum, positive_volume_max=maximum,
                used_point_count=int(used.sum()), omitted_point_count=len(omitted),
                omitted_point_indices=omitted.tolist(), point_coverage_complete=bool(used.all()),
                coordinate_policy="original_no_perturbation", volume_dtype="float64",
                coplanar_diagnostics_available=False)
    return result, meta


def scipy_triangulate(points, chunk_size=65536):
    """Original-coordinate SciPy CPU fallback, without QJ or silent retries."""
    coordinates, input_meta = _prepare_points(points)
    try:
        import scipy
        from scipy.spatial import Delaunay, QhullError
    except ImportError as exc:
        raise TriangulationValidationError("scipy_unavailable", str(exc), input_meta) from exc
    options = "Qbb Qc Qz Q12"
    meta = dict(input_meta, backend="scipy", qhull_options=options, scipy_version=scipy.__version__)
    candidate = coordinates.copy()  # Protect caller storage against backend mutation.
    try:
        topology = Delaunay(candidate, qhull_options=options)
    except QhullError as exc:
        meta["qhull_error"] = str(exc)
        raise TriangulationValidationError("qhull_failure", str(exc), meta) from exc
    if not np.array_equal(candidate, coordinates) or not np.array_equal(topology.points, coordinates):
        _reject("coordinates_changed", "SciPy changed original coordinates", meta)
    try:
        result, validated = validate_tetrahedra(coordinates, topology.simplices,
                                               backend="scipy", chunk_size=chunk_size)
    except TriangulationValidationError as exc:
        exc.metadata.update(qhull_options=options, scipy_version=scipy.__version__)
        raise
    coplanar = np.asarray(topology.coplanar)
    validated.update(qhull_options=options, scipy_version=scipy.__version__,
                     input_dtype=input_meta["input_dtype"], coplanar_diagnostics_available=True,
                     coplanar_point_count=len(coplanar), coplanar_records=coplanar.astype(np.int64).tolist(),
                     coplanar_columns=["point_index", "nearest_facet_index", "nearest_vertex_index"])
    return result, validated
