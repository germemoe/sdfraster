"""Select independent curvature derivatives without resampling Eikonal points."""
import math


def validate_curvature_configuration(curvature_mode, curvature_taps):
    if curvature_mode not in ("inherit", "analytical", "numerical"):
        raise ValueError("curvature_mode must be inherit, analytical or numerical")
    if curvature_taps not in (4, 6):
        raise ValueError("curvature_taps must be 4 or 6")
    return curvature_mode, curvature_taps


def resolve_curvature_settings(
    gradient_mode, gradient_taps, curvature_mode="inherit", curvature_taps=6,
    eps=1e-3,
):
    """Resolve the requested operator; analytical derivatives do not use eps/taps.

    inherit follows both original gradient_mode and gradient_taps, so its default
    preserves numerical4 and numerical6 configurations. curvature_taps applies
    only to explicit numerical curvature. Both routes share the world-space eps
    selected by gradient_eps_mode (fixed or active hash-level resolution).
    """
    validate_curvature_configuration(curvature_mode, curvature_taps)
    if gradient_mode not in ("analytical", "numerical"):
        raise ValueError("gradient_mode must be analytical or numerical")
    mode = gradient_mode if curvature_mode == "inherit" else curvature_mode
    taps = gradient_taps if curvature_mode == "inherit" else curvature_taps
    if mode == "numerical":
        if taps not in (4, 6):
            raise ValueError("Numerical curvature requires 4 or 6 taps")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("Numerical curvature requires a finite positive epsilon")
    return {
        "requested_mode": curvature_mode,
        "requested_taps": curvature_taps,
        "requested_eps": float(eps),
        "effective_mode": mode,
        "effective_taps": taps if mode == "numerical" else None,
        "effective_eps": float(eps) if mode == "numerical" else None,
        "sample_position_policy": (
            "fixed_detached" if curvature_mode != "inherit" else
            "legacy_numerical_connected" if gradient_mode == "numerical" else
            "legacy_analytical_detached"
        ),
    }


def compute_regularization_derivatives(
    model, points, *, gradient_mode="analytical", gradient_taps=4, eps=1e-3,
    hessian_mode="diagonal", curvature_mode="inherit", curvature_taps=6,
    need_gradient=True, curvature_active=True,
):
    """Return (Eikonal/TV gradient, curvature tensor) at exactly the given points.

    The inherit route retains the existing joint call, including its returned
    gradients when only curvature is active. Explicit curvature routes never
    substitute their gradients for a different Eikonal/TV operator. Inactive
    consumers and empty points create no extra derivative graph. Explicit modes
    fix sample positions for every consumer by detaching coordinates once; SDF
    parameter graphs remain live. inherit preserves the original operator's
    position policy (analytical detaches, numerical can keep the input graph).
    """
    settings = resolve_curvature_settings(
        gradient_mode, gradient_taps, curvature_mode, curvature_taps, eps,
    )
    if points.shape[0] == 0 or not (need_gradient or curvature_active):
        return None, None

    if curvature_mode != "inherit":
        points = points.detach()

    same_operator = (
        settings["effective_mode"] == gradient_mode
        and (gradient_mode == "analytical"
             or settings["effective_taps"] == gradient_taps)
    )
    if curvature_mode == "inherit" or (need_gradient and curvature_active and same_operator):
        _, gradients, hessian = model.compute_sdf_gradients(
            points=points, mode=gradient_mode, taps=gradient_taps, eps=eps,
            hessian_mode=hessian_mode, compute_hessian=curvature_active,
        )
        return gradients, hessian

    gradients = None
    hessian = None
    if need_gradient:
        _, gradients, _ = model.compute_sdf_gradients(
            points=points, mode=gradient_mode, taps=gradient_taps, eps=eps,
            hessian_mode=hessian_mode, compute_hessian=False,
        )
    if curvature_active:
        _, _, hessian = model.compute_sdf_gradients(
            points=points, mode=settings["effective_mode"],
            taps=settings["effective_taps"] or curvature_taps, eps=eps,
            hessian_mode=hessian_mode, compute_hessian=True,
        )
    return gradients, hessian
