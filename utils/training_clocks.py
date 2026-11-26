"""Explicit training clocks. No model queries, randomness, or inference state."""
from numbers import Integral


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _int(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return int(value)


def effective_training_steps(config):
    planned = _int(_get(config, "iterations"), "iterations", 1)
    cap = _int(_get(config, "stop_after_steps", 0), "stop_after_steps", 0)
    if cap > planned:
        raise ValueError("stop_after_steps cannot exceed the planned iterations")
    return cap or planned


def resolve_optimizer_lr(freeze_start, lr_horizon=0, lr_mode="legacy_spiking",
                         lr_peak_start=-1, midpoint=2000):
    freeze = _int(freeze_start, "freeze_start", 0)
    requested = _int(lr_horizon, "lr_horizon", 0)
    peak = _int(lr_peak_start, "lr_peak_start", -1)
    midpoint = _int(midpoint, "midpoint", 0)
    if lr_mode not in ("legacy_spiking", "exponential"):
        raise ValueError("lr_mode must be legacy_spiking or exponential")
    return {
        "lr_mode": lr_mode, "requested_lr_horizon": requested,
        "effective_lr_horizon": requested or freeze,
        "requested_lr_peak_start": peak,
        "effective_lr_peak_start": midpoint if peak == -1 else peak,
        "policy_scope": "TetOptimizer; Frozen optimizer policy unchanged",
    }


def resolve_training_clocks(config):
    planned = _int(_get(config, "iterations"), "iterations", 1)
    stop = effective_training_steps(config)
    freeze = _int(_get(config, "freeze_start"), "freeze_start", 0)
    lr = resolve_optimizer_lr(freeze, _get(config, "lr_horizon", 0),
                              _get(config, "lr_mode", "legacy_spiking"),
                              _get(config, "lr_peak_start", -1))
    explicit_lr = (lr["requested_lr_horizon"] != 0 or
                   lr["lr_mode"] != "legacy_spiking" or
                   lr["requested_lr_peak_start"] != -1)
    if explicit_lr and stop > freeze:
        raise ValueError("Explicit TetOptimizer LR controls require stopping at or before "
                         "freeze_start; Frozen LR support is unchanged")
    return {
        "clock_schema_version": 2, "planned_iterations": planned,
        "stop_after_steps": _int(_get(config, "stop_after_steps", 0), "stop_after_steps", 0),
        "training_budget_steps": stop, "geometry_freeze_start": freeze,
        "event_clock_policy": "Explicit event values are unchanged by the stop cap",
        "frozen_stage_lr_horizon": max(planned - freeze, 0), **lr,
    }


def resolve_checkpoint_milestones(spec, clocks):
    if not isinstance(spec, str):
        raise ValueError("checkpoint_milestones must be a comma-separated string")
    planned = {int(x) for x in spec.split(",") if x.strip()}
    horizon = clocks["planned_iterations"]
    stop = clocks["training_budget_steps"]
    if any(step <= 0 or step > horizon for step in planned):
        raise ValueError("checkpoint milestones must be within 1..iterations")
    effective = {step for step in planned if step <= stop}
    if stop < horizon:
        effective.add(stop)
    return planned, effective


def resolve_curvature_weight_clock(config):
    start = _int(_get(config, "curvature_start"), "curvature_start", None)
    end = _int(_get(config, "curvature_end"), "curvature_end", None)
    warmup = _int(_get(config, "curvature_warmup"), "curvature_warmup", 0)
    origin = _int(_get(config, "curvature_weight_origin", -1),
                  "curvature_weight_origin", -1)
    requested_end = _int(_get(config, "curvature_weight_end", -1),
                         "curvature_weight_end", -1)
    weight_end = end if requested_end == -1 else requested_end
    if origin >= 0 and weight_end < origin:
        raise ValueError("curvature_weight_end cannot precede explicit weight origin")
    decay = _get(config, "curvature_decay_enabled", True)
    if not isinstance(decay, bool):
        raise ValueError("curvature_decay_enabled must be a boolean")
    return {
        "active_start": start, "active_end": end,
        "requested_weight_origin": origin, "requested_weight_end": requested_end,
        "weight_mode": "legacy_start_coupled" if origin == -1 else "fixed_origin",
        "effective_weight_origin": 0 if origin == -1 else origin,
        "effective_warmup_end": start + warmup if origin == -1 else origin + warmup,
        "effective_weight_end": weight_end,
        "warmup_length": warmup,
        "decay_enabled": decay,
        "gate_semantics": "start <= zero-based iteration < end",
    }


def curvature_weight_factor(backbone, iteration, clock):
    if clock["weight_mode"] == "legacy_start_coupled":
        # Preserve the original argument values and floating-point operations.
        return backbone.get_curvature_weight_decay(
            iteration=iteration, warmup_end=clock["effective_warmup_end"],
            decay_end=clock["effective_weight_end"], decay_enabled=clock["decay_enabled"])
    origin = clock["effective_weight_origin"]
    return backbone.get_curvature_weight_decay(
        iteration=max(iteration - origin, 0), warmup_end=clock["warmup_length"],
        decay_end=clock["effective_weight_end"] - origin,
        decay_enabled=clock["decay_enabled"])


def training_clock_metadata(clocks, completed_steps):
    completed = _int(completed_steps, "completed_steps", 0)
    if completed > clocks["training_budget_steps"]:
        raise ValueError("completed_steps exceeds the actual training budget")
    return {**clocks, "completed_steps": completed,
            "lr_horizon": clocks["effective_lr_horizon"],
            "stop_cap_reached": completed == clocks["training_budget_steps"],
            "checkpoint_load_policy": "model_weights_only; optimizer and iteration restart from zero"}
