"""Validated triangulation and crash containment; coordinates are never perturbed."""
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
from utils.delaunay_isolation import NativeDelaunayClient, NativeTriangulationError, is_external_termination
from utils.triangulation_validation import (validate_tetrahedra, scipy_triangulate,
                                            TriangulationValidationError)

BACKENDS = ("isolated_native", "legacy_native", "scipy")

def validate_triangulation_configuration(backend, timeout_seconds):
    if backend not in BACKENDS:
        raise ValueError("triangulation_backend must be isolated_native, legacy_native or scipy")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("triangulation_timeout_seconds must be finite and positive")

class TriangulationBackend:
    def __init__(self, failure_dir, backend="isolated_native", timeout_seconds=120,
                 client_factory=NativeDelaunayClient):
        validate_triangulation_configuration(backend, timeout_seconds)
        self.failure_dir = Path(failure_dir)
        self.backend = backend
        self.timeout_seconds = float(timeout_seconds)
        self.client_factory = client_factory
        self.client = None
        self.last_metadata = None
        self.call_count = 0
        self.fallback_count = 0

    def _client(self):
        if self.client is None:
            # Three million input points and their ~18M tetrahedra fit this bound.
            self.client = self.client_factory(self.failure_dir, self.timeout_seconds,
                                               max_payload_bytes=1024**3)
        return self.client

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def _record(self, metadata):
        self.last_metadata = metadata
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        with (self.failure_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(metadata, allow_nan=False) + "\n")

    def compute(self, points, *, high_precision=False, context=None):
        started = time.perf_counter()
        points = np.ascontiguousarray(points, dtype=np.float64)
        self.call_count += 1
        meta = {"requested_backend": self.backend, "high_precision_override": bool(high_precision),
                "call_number": self.call_count, "context": context,
                "point_count": int(points.shape[0]), "input_coordinates_changed": False}
        try:
            if high_precision or self.backend == "scipy":
                indices, validation = scipy_triangulate(points)
                meta.update(actual_backend="scipy", validation=validation, fallback=False)
            elif self.backend == "legacy_native":
                # Explicit compatibility option: fatal CUDA assertions remain in this process.
                import torch
                from gdel3d import Del
                raw, previous = Del(len(points)).compute(torch.from_numpy(points))
                torch.cuda.synchronize()
                indices, validation = validate_tetrahedra(points, raw.numpy(), backend="native")
                del previous
                meta.update(actual_backend="legacy_native", validation=validation, fallback=False)
            else:
                client = self._client()
                try:
                    raw, native_meta = client.compute(points, context=context)
                    try:
                        indices, validation = validate_tetrahedra(points, raw, backend="native")
                    except TriangulationValidationError as error:
                        native_meta.update(status="native_topology_invalid", error=str(error),
                                           topology_validation_error=error.metadata)
                        try:
                            captured = client.capture_failure(points, native_meta)
                        except Exception as save_error:
                            if is_external_termination(save_error):
                                raise
                            captured = {**native_meta, "capture_error": str(save_error)}
                        finally:
                            self.close()
                        raise NativeTriangulationError("Native topology failed geometry validation", captured) from error
                    meta.update(actual_backend="isolated_native", native=native_meta,
                                validation=validation, fallback=False)
                except NativeTriangulationError as error:
                    # Reject an external cancellation wrapped by an older/custom client.
                    if is_external_termination(error.__cause__):
                        raise error.__cause__
                    self.fallback_count += 1
                    meta.update(native_failure=error.metadata, fallback=True)
                    indices, validation = scipy_triangulate(points)
                    meta.update(actual_backend="scipy", validation=validation)
            meta.update(status="validated", tetrahedron_count=int(len(indices)),
                        fallback_count=self.fallback_count, elapsed_seconds=time.perf_counter()-started)
            self._record(meta)
            return indices, meta
        except Exception as error:
            if is_external_termination(error):
                self.close()
                raise
            meta.update(status="failed_closed", error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.perf_counter()-started)
            if hasattr(error, "metadata"):
                meta["validation_error"] = error.metadata
            # A failed fallback must also preserve the exact input, with a distinct request id.
            if "native_failure" not in meta:
                capture = {"request_id": uuid.uuid4().hex, "status": "failed_closed",
                           "error_type": type(error).__name__, "error": str(error),
                           "request_context": context}
                try:
                    meta["failure_capture"] = self._client().capture_failure(points, capture)
                except Exception as save_error:
                    if is_external_termination(save_error):
                        raise
                    meta["capture_error"] = str(save_error)
            self._record(meta)
            raise
