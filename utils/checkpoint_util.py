"""Persist inference-relevant state that is absent from model.state_dict()."""
import json
from pathlib import Path
from utils.encoding_precision import validate_encoding_precision
from utils.matmul_precision import (matmul_precision_state,
                                    configure_float32_matmul_precision,
                                    validate_float32_matmul_precision)


def save_runtime_state(model, path):
    state = {
        "schema_version": 1,
        "model_class": type(model).__name__,
        "frozen": bool(model.frozen),
        "tet_normal_mode": getattr(model, "tet_normal_mode", "raw"),
        **matmul_precision_state(model),
    }
    if hasattr(model, "triangulation_backend"):
        state["triangulation_backend"] = model.triangulation_backend
        state["triangulation_timeout_seconds"] = model.triangulation_timeout_seconds
        state["triangulation_last_metadata"] = getattr(model, "triangulation_last_metadata", None)
    if hasattr(model, "current_sh_deg"):
        state["current_sh_deg"] = int(model.current_sh_deg)
    if hasattr(model, "backbone") and hasattr(model.backbone, "active_levels"):
        state["active_hash_levels"] = int(model.backbone.active_levels)
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "encoding_precision"):
        state["encoding_precision"] = validate_encoding_precision(backbone.encoding_precision)
        state["encoding_backend"] = backbone.encoding_backend
        encoding = backbone.encoding
        dtype = getattr(encoding, "dtype", None)
        if dtype is None:
            parameter = next(encoding.parameters(), None)
            dtype = parameter.dtype if parameter is not None else None
        if dtype is not None:
            state["encoding_dtype"] = str(dtype).removeprefix("torch.")
    Path(path).write_text(json.dumps(state, indent=2))
    return state


def restore_runtime_state(model, path):
    path = Path(path)
    if not path.exists():
        return None  # Keep legacy checkpoints readable.
    state = json.loads(path.read_text())
    if state.get("schema_version") != 1:
        raise ValueError("Unknown checkpoint runtime-state version")
    if state.get("frozen"):
        raise ValueError("Frozen-model checkpoint requires its matching model loader")
    if "encoding_precision" in state:
        precision = validate_encoding_precision(state["encoding_precision"])
        backbone = model.backbone
        if precision != getattr(backbone, "encoding_precision", "float16"):
            raise ValueError("Checkpoint encoding precision differs from constructed model")
        if state.get("encoding_backend", backbone.encoding_backend) != backbone.encoding_backend:
            raise ValueError("Checkpoint encoding backend differs from constructed model")
        if backbone.encoding_backend == "tcnn" and "encoding_dtype" in state:
            actual_dtype = str(backbone.encoding.dtype).removeprefix("torch.")
            if state["encoding_dtype"] != actual_dtype:
                raise ValueError("Checkpoint native encoding dtype differs from constructed model")
    if "float32_matmul_precision" in state:
        precision = validate_float32_matmul_precision(state["float32_matmul_precision"])
        backbone = getattr(model, "backbone", None)
        declared = getattr(backbone, "float32_matmul_precision", None)
        if declared is not None and precision != validate_float32_matmul_precision(declared):
            raise ValueError("Checkpoint float32 matmul precision differs from constructed model")
        # Restore process-global mode even if other code changed it after construction.
        actual = configure_float32_matmul_precision(precision)
        if ("cuda_matmul_allow_tf32" in state and
                state["cuda_matmul_allow_tf32"] != actual["cuda_matmul_allow_tf32"]):
            raise ValueError("Checkpoint CUDA matmul TF32 flag contradicts its math mode")
    degree = state["current_sh_deg"]
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= model.max_sh_deg:
        raise ValueError("Invalid checkpoint SH degree")
    mode = state["tet_normal_mode"]
    if mode not in ("raw", "unit"):
        raise ValueError("Invalid checkpoint tetrahedron normal mode")
    model.current_sh_deg = degree
    model.tet_normal_mode = mode
    if "active_hash_levels" in state:
        levels = state["active_hash_levels"]
        if isinstance(levels, bool) or not isinstance(levels, int) or not 1 <= levels <= model.backbone.L:
            raise ValueError("Invalid checkpoint hash-grid active levels")
        model.backbone.active_levels = levels
    return state
