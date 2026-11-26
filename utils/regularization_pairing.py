"""Training-only requirement: active curvature is covered by active Eikonal."""
import math
from numbers import Real
from utils.training_clocks import effective_training_steps


def _value(config, name):
    return config[name] if isinstance(config, dict) else getattr(config, name)


def _integer(config, name):
    value = _value(config, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _weight(config, name):
    value = _value(config, name)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _active_steps(weight, start, end, iterations):
    """Inclusive executed step bounds for start <= iteration < end."""
    if weight <= 0:
        return None
    first = max(0, start)
    last = min(iterations - 1, end - 1)
    return [first, last] if first <= last else None


def validate_regularization_pairing(config):
    """Reject training curvature without simultaneous Eikonal on every step.

    This validation never changes weights or schedules and is intentionally
    called by training/plan readiness, not by derivative or diagnostic methods.
    Future/empty intervals outside the executed horizon require no pairing.
    """
    iterations = effective_training_steps(config)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    curv_weight = _weight(config, "lambda_curvature")
    eik_weight = _weight(config, "lambda_eikonal")
    curv_start = _integer(config, "curvature_start")
    curv_end = _integer(config, "curvature_end")
    eik_start = _integer(config, "eikonal_start")
    eik_end = _integer(config, "eikonal_end")
    curvature = _active_steps(curv_weight, curv_start, curv_end, iterations)
    eikonal = _active_steps(eik_weight, eik_start, eik_end, iterations)
    metadata = {
        "constraint": "active_curvature_requires_simultaneous_eikonal",
        "schedule_semantics": "start_inclusive_end_exclusive_zero_based_iteration",
        "executed_iteration_range": [0, iterations - 1],
        "curvature_active_steps": curvature,
        "eikonal_active_steps": eikonal,
        "required": curvature is not None,
    }
    if curvature is None:
        return metadata
    if eikonal is None or eikonal[0] > curvature[0] or eikonal[1] < curvature[1]:
        raise ValueError(
            "Curvature training requires positive lambda_eikonal and an Eikonal "
            "schedule covering every active curvature iteration. "
            f"Executed curvature steps={curvature}, Eikonal steps={eikonal}; "
            f"lambda_curvature={curv_weight}, lambda_eikonal={eik_weight}. "
            "Intervals use start <= iteration < end and are clipped to "
            f"[0, {iterations - 1}]. Change the explicit experiment configuration; "
            "weights and schedules were not adjusted."
        )
    return metadata
