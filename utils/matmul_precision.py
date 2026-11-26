"""Explicit process-global float32 GEMM math mode; independent of tensor dtype.

One active model per process is assumed. Do not interleave models configured
with different math modes. No CUDA context or model import is needed here.
"""
import json
from pathlib import Path
import torch

FLOAT32_MATMUL_PRECISIONS = ("high", "highest")


def validate_float32_matmul_precision(value):
    if value not in FLOAT32_MATMUL_PRECISIONS:
        raise ValueError("float32_matmul_precision must be high or highest")
    return value


def matmul_precision_state(model=None):
    """Read actual global mode, rejecting a model/config that would mislabel it."""
    actual = validate_float32_matmul_precision(torch.get_float32_matmul_precision())
    backbone = getattr(model, "backbone", None)
    declared = getattr(backbone, "float32_matmul_precision", None)
    if declared is not None and validate_float32_matmul_precision(declared) != actual:
        raise ValueError(
            "Process float32 matmul precision differs from the active model. "
            "Use one active math mode per process; do not relabel a checkpoint."
        )
    return {
        "float32_matmul_precision": actual,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        # Diagnostic only: this option never changes cuDNN convolution precision.
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def configure_float32_matmul_precision(value):
    """Apply after model-module imports, which retain their legacy high default."""
    value = validate_float32_matmul_precision(value)
    torch.set_float32_matmul_precision(value)
    state = matmul_precision_state()
    if state["float32_matmul_precision"] != value:
        raise RuntimeError("PyTorch did not apply requested float32 matmul precision")
    return state


def float32_matmul_precision_is_explicit(args, argv):
    name = "--float32_matmul_precision"
    if any(token == name or token.startswith(name + "=") for token in argv):
        return True
    path = getattr(args, "config", None)
    return path is not None and "float32_matmul_precision" in json.loads(Path(path).read_text())


def reconcile_checkpoint_matmul_precision(args, model, argv):
    """Keep source checkpoint math mode; --ckpt is weights/structure only."""
    requested = validate_float32_matmul_precision(args.float32_matmul_precision)
    checkpoint = matmul_precision_state(model)["float32_matmul_precision"]
    explicit = float32_matmul_precision_is_explicit(args, argv)
    if explicit and requested != checkpoint:
        raise ValueError(
            f"Requested float32_matmul_precision={requested} differs from "
            f"checkpoint actual math mode={checkpoint}. --ckpt preserves the "
            "source checkpoint math mode; start a fresh run to change it. "
            "This is not a full training resume."
        )
    args.float32_matmul_precision = checkpoint
    return {
        "requested": requested, "checkpoint": checkpoint, "explicit": bool(explicit),
        "policy": "source_checkpoint_actual_math_mode; model_weights_and_structure_only",
    }


def checkpoint_float32_matmul_precision(path):
    """Read source math mode before camera/data setup; verify saved metadata."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    mode = validate_float32_matmul_precision(config.get("float32_matmul_precision", "high"))
    runtime = path / "runtime_state.json"
    if runtime.exists():
        state = json.loads(runtime.read_text())
        if "float32_matmul_precision" in state:
            actual = validate_float32_matmul_precision(state["float32_matmul_precision"])
            if actual != mode:
                raise ValueError("Checkpoint config and actual float32 matmul precision disagree")
            if "cuda_matmul_allow_tf32" in state:
                flag = state["cuda_matmul_allow_tf32"]
                if not isinstance(flag, bool) or flag != (mode == "high"):
                    raise ValueError("Checkpoint CUDA matmul TF32 flag contradicts its math mode")
    return mode
