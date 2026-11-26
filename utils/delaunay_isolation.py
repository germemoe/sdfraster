"""Persistent exec-only isolation for fatal gDel CUDA/native assertions.

The caller retains candidate coordinates and validates returned *raw* topology
before changing its model. This module never imports torch, gDel, Model or tcnn.
Only failed calls write candidate arrays. One client serializes its requests;
use an explicit close/context manager (the finalizer is an additional guard).
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
import weakref
from multiprocessing import Pipe

import numpy as np

PROTOCOL_VERSION = 1
HEADER_MAX_BYTES = 32768
DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
PARENT_PID_ENV = "SDFRASTER_DELAUNAY_EXPECTED_PARENT_PID"


class NativeTriangulationError(RuntimeError):
    """Native execution/transport failed; the owned worker has been reclaimed."""

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata
        self.failure_dir = metadata.get("failure_dir")


def is_external_termination(error):
    """Recognize actual caller cancellation, without importing a training policy."""
    if isinstance(error, InterruptedError):
        return True
    budget = sys.modules.get("training_time_budget")
    budget_error = getattr(budget, "BudgetExceeded", None)
    return isinstance(budget_error, type) and isinstance(error, budget_error)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def send_header(connection, value):
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf8")
    if len(encoded) > HEADER_MAX_BYTES:
        raise ValueError("IPC header exceeds bounded size")
    connection.send_bytes(encoded)


def receive_header(connection):
    encoded = connection.recv_bytes(maxlength=HEADER_MAX_BYTES)
    def reject_nonfinite(token):
        raise ValueError("Nonfinite JSON constant in IPC header: " + token)
    value = json.loads(encoded.decode("utf8"), parse_constant=reject_nonfinite)
    if (not isinstance(value, dict) or type(value.get("protocol")) is not int
            or value.get("protocol") != PROTOCOL_VERSION):
        raise ValueError("Invalid IPC protocol/header")
    return value


def checked_shape(header, width, dtype, limit):
    shape = header.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(x) is not int or x < 0 for x in shape)
            or shape[1] != width or header.get("dtype") != dtype):
        raise ValueError("Invalid IPC shape/dtype")
    byte_count = shape[0] * width * np.dtype(dtype).itemsize
    if type(header.get("payload_bytes")) is not int or header["payload_bytes"] != byte_count:
        raise ValueError("IPC payload size disagrees with shape/dtype")
    if byte_count > limit:
        raise ValueError("IPC payload exceeds bounded size")
    return tuple(shape), byte_count


def _cleanup_state(state):
    """Kill only our start_new_session process group, then reap its leader."""
    process = state.pop("process", None)
    connection = state.pop("connection", None)
    io_thread = state.pop("io_thread", None)
    report = {"worker_reaped": True, "process_group_terminated": True}
    if process is not None:
        # The leader is not polled/reaped before group termination: its PID
        # remains reserved while even a crashed worker's descendants are killed.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            if sig == signal.SIGTERM:
                time.sleep(0.05)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            report["worker_reaped"] = False
        report.update(worker_pid=process.pid, worker_returncode=process.returncode)
    if io_thread is not None and io_thread is not threading.current_thread():
        io_thread.join(timeout=3)
        report["ipc_thread_stopped"] = not io_thread.is_alive()
    if connection is not None:
        connection.close()
    return report


class NativeDelaunayClient:
    """compute(float64 Nx3) -> (raw int32 Mx4, diagnostic metadata).

    A fresh executable process (never a Python multiprocessing fork) owns all
    native CUDA calls. timeout_seconds bounds startup handshake + request I/O +
    compute. worker_command is an explicit trusted fixture/alternative worker
    argv for CPU transport tests; no shell is used. On native errors all workers
    are replaced, including non-CUDA Python errors.
    """

    def __init__(self, failure_dir, timeout_seconds=120,
                 max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES, worker_command=None):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(max_payload_bytes) is not int or max_payload_bytes < 96:
            raise ValueError("max_payload_bytes must be an integer >= 96")
        if worker_command is not None and (
                not isinstance(worker_command, (list, tuple)) or not worker_command
                or any(not isinstance(x, str) for x in worker_command)):
            raise ValueError("worker_command must be a nonempty argv list")
        self.failure_root = Path(failure_dir)
        self.timeout_seconds = float(timeout_seconds)
        self.max_payload_bytes = max_payload_bytes
        self.worker_command = list(worker_command) if worker_command is not None else None
        self._state = {}
        self._lock = threading.RLock()
        self._closed = False
        self._worker_number = 0
        self._request_number = 0
        self._worker_info = {}
        self._source = {
            "client_path": str(Path(__file__).resolve()),
            "client_sha256": sha256_file(__file__),
            "worker_path": str(Path(__file__).with_name("delaunay_worker.py").resolve()),
            "worker_sha256": sha256_file(Path(__file__).with_name("delaunay_worker.py")),
        }
        self._finalizer = weakref.finalize(self, _cleanup_state, self._state)

    @property
    def worker_pid(self):
        process = self._state.get("process")
        return process.pid if process is not None else None

    def _start_worker(self):
        self.failure_root.mkdir(parents=True, exist_ok=True)
        self._worker_number += 1
        self._worker_info = {}
        identity = uuid.uuid4().hex
        logfile = self.failure_root / f"worker_{identity}.native.log"
        parent_connection, child_connection = Pipe(duplex=True)
        command = (self.worker_command or [
            sys.executable, "-u", self._source["worker_path"]
        ]) + ["--connection-fd", str(child_connection.fileno()),
              "--max-payload-bytes", str(self.max_payload_bytes)]
        try:
            environment = os.environ.copy()
            environment[PARENT_PID_ENV] = str(os.getpid())
            with logfile.open("xb") as log:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    pass_fds=(child_connection.fileno(),), start_new_session=True,
                    close_fds=True, env=environment,
                )
            self._state.update(process=process, connection=parent_connection)
            self._worker_info = {
                "worker_pid": process.pid, "worker_process_group": process.pid,
                "worker_generation": self._worker_number,
                "native_log": str(logfile), "worker_argv": command,
                "process_creation": "subprocess_exec_start_new_session_no_python_fork",
                "expected_parent_pid": os.getpid(),
                "source": dict(self._source),
                "worker_command_sources": [
                    {"path": str(Path(token).resolve()), "sha256": sha256_file(token)}
                    for token in command if token.endswith(".py") and Path(token).is_file()],
            }
        except BaseException:
            parent_connection.close()
            raise
        finally:
            # Keeping this fd in the parent would suppress EOF after worker death.
            child_connection.close()

    def capture_failure(self, points, metadata):
        """Persist an exact candidate for native or caller topology failures.

        No backend is retried here; the caller may add its validation diagnostic.
        This method deliberately does not import or couple to topology validation.
        """
        directory = self.failure_root / ("failure_" + uuid.uuid4().hex)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "candidate.npy"
        np.save(path, points, allow_pickle=False)
        record = dict(metadata)
        log_path = record.get("native_log")
        if log_path and Path(log_path).is_file():
            record.update(native_log_sha256=sha256_file(log_path),
                          native_log_bytes=Path(log_path).stat().st_size)
        record.update(
            failure_dir=str(directory), candidate_path=str(path),
            candidate_sha256=sha256_file(path),
            coordinate_bytes_sha256=hashlib.sha256(points.tobytes(order="C")).hexdigest(),
            candidate_shape=list(points.shape), candidate_dtype=points.dtype.str,
            candidate_nbytes=int(points.nbytes), source=dict(self._source),
        )
        (directory / "failure.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n")
        return record

    def compute(self, points, *, context=None):
        if not isinstance(points, np.ndarray) or points.dtype != np.dtype(np.float64):
            raise ValueError("candidate must be a numpy native float64 array")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
            raise ValueError("candidate must have shape (N>=4, 3)")
        if points.shape[0] >= np.iinfo(np.int32).max or points.nbytes > self.max_payload_bytes:
            raise ValueError("candidate exceeds bounded input/index size")
        if not np.isfinite(points).all():
            raise ValueError("candidate coordinates must be finite")
        # Own a snapshot: native code and concurrent caller mutation cannot alter
        # the exact points retained for failure capture/parent topology checks.
        candidate = np.array(points, dtype=np.float64, order="C", copy=True)
        if context is not None:
            context = json.loads(json.dumps(context, allow_nan=False))
            if len(json.dumps(context).encode()) > HEADER_MAX_BYTES // 2:
                raise ValueError("request context exceeds bounded metadata size")
        with self._lock:
            if self._closed:
                raise RuntimeError("NativeDelaunayClient is closed")
            started = time.monotonic()
            self._request_number += 1
            request_id = uuid.uuid4().hex
            request = {
                "protocol": PROTOCOL_VERSION, "op": "compute", "request_id": request_id,
                "shape": list(candidate.shape), "dtype": candidate.dtype.str,
                "payload_bytes": int(candidate.nbytes),
            }
            metadata = {"request_id": request_id, "request_number": self._request_number,
                        "request_started_utc": datetime.now(timezone.utc).isoformat(),
                        "request_context": context, "timeout_seconds": self.timeout_seconds,
                        "max_payload_bytes": self.max_payload_bytes}
            result = {}
            done = threading.Event()
            try:
                new_worker = not self._state
                if new_worker:
                    self._start_worker()
                metadata.update(self._worker_info)
                connection = self._state["connection"]
                expected_pid = self.worker_pid

                def exchange():
                    try:
                        if new_worker:
                            result["ipc_stage"] = "startup_header"
                            hello = receive_header(connection)
                            if (hello.get("op") != "ready" or hello.get("worker_pid") != expected_pid
                                    or not isinstance(hello.get("metadata"), dict)):
                                raise ValueError("Invalid worker startup schema/pid")
                            result["hello"] = hello["metadata"]
                        result["ipc_stage"] = "input_header"
                        send_header(connection, request)
                        result["ipc_stage"] = "input_payload"
                        connection.send_bytes(candidate.tobytes(order="C"))
                        result["ipc_stage"] = "response_header"
                        header = receive_header(connection)
                        if header.get("request_id") != request_id:
                            raise ValueError("IPC response request_id mismatch")
                        if not isinstance(header.get("metadata"), dict):
                            raise ValueError("Invalid response diagnostic metadata")
                        result["response_metadata"] = header["metadata"]
                        if header.get("op") == "error":
                            raise RuntimeError("worker_exception: " + str(header.get("error", "")))
                        if header.get("op") != "result":
                            raise ValueError("Invalid IPC result opcode")
                        shape, byte_count = checked_shape(
                            header, 4, np.dtype(np.int32).str, self.max_payload_bytes)
                        result["ipc_stage"] = "result_payload"
                        payload = connection.recv_bytes(maxlength=byte_count)
                        if len(payload) != byte_count:
                            raise ValueError("IPC result bytes disagree with announced shape")
                        result["indices"] = np.frombuffer(payload, dtype=np.int32).reshape(shape).copy()
                    except BaseException as error:
                        result["error"] = error
                    finally:
                        done.set()

                thread = threading.Thread(target=exchange, name="gdel-ipc", daemon=True)
                self._state["io_thread"] = thread
                thread.start()
                remaining = max(0, self.timeout_seconds - (time.monotonic() - started))
                if not done.wait(remaining):
                    raise TimeoutError("Native triangulation request timed out")
                thread.join()
                self._state.pop("io_thread", None)
                if "hello" in result:
                    self._worker_info["worker_startup_metadata"] = result["hello"]
                    metadata["worker_startup_metadata"] = result["hello"]
                metadata["worker_response_metadata"] = result.get("response_metadata", {})
                if "error" in result:
                    raise result["error"]
                metadata.update(elapsed_seconds=time.monotonic() - started,
                                raw_tetrahedra=int(result["indices"].shape[0]),
                                raw_dtype=result["indices"].dtype.str,
                                status="native_success", topology_validated=False)
                return result["indices"], metadata
            except BaseException as error:
                metadata["worker_response_metadata"] = result.get("response_metadata", {})
                metadata.update(self._worker_info)
                if "hello" in result:
                    metadata["worker_startup_metadata"] = result["hello"]
                metadata.update(
                    status="native_failure", ipc_stage=result.get("ipc_stage", "process_spawn"),
                    error_type=type(error).__name__,
                    error=str(error), elapsed_seconds=time.monotonic() - started)
                metadata["cleanup"] = _cleanup_state(self._state)
                try:
                    metadata = self.capture_failure(candidate, metadata)
                except Exception as save_error:
                    if is_external_termination(save_error):
                        raise
                    metadata["capture_error"] = f"{type(save_error).__name__}: {save_error}"
                if not isinstance(error, Exception) or is_external_termination(error):
                    raise
                raise NativeTriangulationError(
                    f"Native triangulation failed ({type(error).__name__}); "
                    f"worker reclaimed; diagnostics={metadata.get('failure_dir')}", metadata
                ) from error

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
