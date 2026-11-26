"""Exec-only gDel worker; dedicated Connection IPC and native log fds.

Do not add model/tcnn imports. A CUDA context belongs only to this process.
Every exception ends the worker; the parent must reap and reconstruct it.
"""
# This guard intentionally precedes NumPy/torch/native imports and heavy setup.
import ctypes
import os
import signal
import sys

PARENT_PID_ENV = "SDFRASTER_DELAUNAY_EXPECTED_PARENT_PID"


def arm_parent_death_signal(expected_parent_pid=None):
    """Linux self-only SIGKILL when our creating parent dies.

    No preexec_fn is used (unsafe with an initialized multi-threaded CUDA parent).
    The parent PID check closes the fork/exec-to-prctl race: if already orphaned,
    kill this worker before any native compute. Alternate trusted workers/CPU
    fixtures must explicitly call this hook and include the returned metadata.
    """
    if sys.platform != "linux":
        raise RuntimeError("native process isolation requires Linux PDEATHSIG")
    value = (os.environ.get(PARENT_PID_ENV)
             if expected_parent_pid is None else expected_parent_pid)
    try:
        expected = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("missing/invalid dedicated expected-parent PID") from error
    # A container training entry may legitimately be its PID namespace init.
    # Keep the post-prctl getppid race check for PID1 exactly as for other parents.
    if expected < 1:
        raise RuntimeError("expected-parent PID must be positive")
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG=1, applies to this process only; never signal another PID.
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "prctl(PR_SET_PDEATHSIG) failed")
    actual = os.getppid()
    if actual != expected:
        os.kill(os.getpid(), signal.SIGKILL)
        raise RuntimeError("worker survived self-SIGKILL unexpectedly")
    return {"enabled": True, "signal": "SIGKILL", "expected_parent_pid": expected,
            "observed_parent_pid": actual, "scope": "worker_self_linux_prctl"}


_PARENT_DEATH_GUARD = (
    arm_parent_death_signal() if __name__ == "__main__" else None)

import argparse
import importlib.util
import resource
from pathlib import Path
import time
import traceback
from multiprocessing.connection import Connection

import numpy as np

# Direct script execution uses utils/ as sys.path[0], without package imports.
try:
    from .delaunay_isolation import (PROTOCOL_VERSION, checked_shape,
                                    receive_header, send_header, sha256_file)
except ImportError:
    from delaunay_isolation import (PROTOCOL_VERSION, checked_shape,
                                   receive_header, send_header, sha256_file)


_NATIVE_INFO = None


def native_compute(points):
    """Use the existing float64 native binary; never modify/ignore native asserts."""
    import torch
    import gdel3d
    if gdel3d.get_dtype() != "float64":
        raise RuntimeError("gDel native binary must accept float64 coordinates")
    global _NATIVE_INFO
    if _NATIVE_INFO is None:
        _NATIVE_INFO = {
            "native_module_path": str(Path(gdel3d.__file__).resolve()),
            "native_module_sha256": sha256_file(gdel3d.__file__),
            "native_input_dtype": gdel3d.get_dtype(),
            "torch_version": torch.__version__,
        }
    native = dict(_NATIVE_INFO)
    started = time.monotonic()
    delaunay = gdel3d.Del(len(points))
    raw, output = delaunay.compute(torch.from_numpy(points))
    # Expose latent CUDA failures inside this process, before sending success.
    torch.cuda.synchronize()
    indices = raw.detach().cpu().numpy().copy(order="C")
    del output, raw, delaunay
    torch.cuda.synchronize()
    if indices.dtype != np.dtype(np.int32):
        raise ValueError("native raw index dtype must be int32")
    native.update(native_compute_seconds=time.monotonic() - started,
                  cuda_synchronized=True, backend="gdel3d_raw")
    return indices, native


def serve(connection, compute, max_payload_bytes, startup_metadata=None):
    """Bounded protocol server; compute injection is only for CPU transport tests."""
    send_header(connection, {
        "protocol": PROTOCOL_VERSION, "op": "ready", "worker_pid": os.getpid(),
        "metadata": startup_metadata or {},
    })
    while True:
        request_id = None
        try:
            header = receive_header(connection)
            request_id = header.get("request_id")
            if header.get("op") != "compute" or not isinstance(request_id, str):
                raise ValueError("Invalid IPC request schema")
            shape, byte_count = checked_shape(
                header, 3, np.dtype(np.float64).str, max_payload_bytes)
            if shape[0] < 4 or shape[0] >= np.iinfo(np.int32).max:
                raise ValueError("Invalid candidate count")
            payload = connection.recv_bytes(maxlength=byte_count)
            if len(payload) != byte_count:
                raise ValueError("IPC candidate bytes disagree with shape")
            points = np.frombuffer(payload, dtype=np.float64).reshape(shape).copy()
            if not np.isfinite(points).all():
                raise ValueError("candidate coordinates must be finite")
            raw, metadata = compute(points)
            if (not isinstance(raw, np.ndarray) or raw.dtype != np.dtype(np.int32)
                    or raw.ndim != 2 or raw.shape[1] != 4):
                raise ValueError("Invalid native raw index shape/dtype")
            raw = np.ascontiguousarray(raw)
            if raw.nbytes > max_payload_bytes:
                raise ValueError("native raw index payload exceeds bounded size")
            if not isinstance(metadata, dict):
                raise ValueError("Invalid compute metadata")
            send_header(connection, {
                "protocol": PROTOCOL_VERSION, "op": "result", "request_id": request_id,
                "shape": list(raw.shape), "dtype": raw.dtype.str,
                "payload_bytes": int(raw.nbytes), "metadata": metadata,
            })
            connection.send_bytes(raw.tobytes(order="C"))
        except EOFError:
            return 0
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            try:
                send_header(connection, {
                    "protocol": PROTOCOL_VERSION, "op": "error", "request_id": request_id,
                    "error": f"{type(error).__name__}: {error}"[:8192],
                    "metadata": {"worker_error_type": type(error).__name__,
                                 "worker_exits_after_exception": True},
                })
            except Exception:
                pass
            return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-fd", type=int, required=True)
    parser.add_argument("--max-payload-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.connection_fd < 0 or args.max_payload_bytes < 96:
        parser.error("invalid IPC fd/payload bound")
    # Native SIGABRT must not create multi-GB CUDA core files on CPFS.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    connection = Connection(args.connection_fd)
    native_spec = importlib.util.find_spec("gdel3d")
    startup = {
        "worker_path": str(Path(__file__).resolve()),
        "worker_sha256": sha256_file(__file__),
        "native_module_path": native_spec.origin if native_spec else None,
            "native_module_sha256": (
            sha256_file(native_spec.origin)
            if native_spec and native_spec.origin and Path(native_spec.origin).is_file() else None),
        "imports_model_or_tcnn": False, "core_dumps_disabled": True,
        "parent_death_guard": _PARENT_DEATH_GUARD,
    }
    try:
        return serve(connection, native_compute, args.max_payload_bytes, startup)
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
