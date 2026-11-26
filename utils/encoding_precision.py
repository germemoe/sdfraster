"""TCNN encoding precision configuration; no CUDA imports or execution."""
import json

ENCODING_PRECISIONS = ("float16", "float32")


def validate_encoding_precision(value):
    if value not in ENCODING_PRECISIONS:
        raise ValueError("encoding_precision must be float16 or float32")
    return value


def encoding_precision_is_explicit(args, argv):
    if any(token == "--encoding_precision" or token.startswith("--encoding_precision=")
           for token in argv):
        return True
    path = getattr(args, "config", None)
    return path is not None and "encoding_precision" in json.loads(path.read_text())


def reconcile_checkpoint_precision(args, model, argv):
    """Keep the loaded native module and the newly saved config consistent.

    --ckpt loads model weights/structure only, without optimizer/RNG/schedule
    resumption. A precision change needs fresh construction, never relabeling an
    already constructed native module.
    """
    requested = validate_encoding_precision(args.encoding_precision)
    checkpoint = validate_encoding_precision(model.backbone.encoding_precision)
    explicit = encoding_precision_is_explicit(args, argv)
    if explicit and requested != checkpoint:
        raise ValueError(
            f"Requested encoding_precision={requested} differs from checkpoint "
            f"model precision={checkpoint}. --ckpt loads the source checkpoint "
            "model weights/structure with its precision; start a fresh run to "
            "change encoding precision. This is not a full training resume."
        )
    args.encoding_precision = checkpoint
    return {"requested": requested, "checkpoint": checkpoint,
            "explicit": bool(explicit),
            "policy": "source_checkpoint_model_precision; model_weights_and_structure_only"}
