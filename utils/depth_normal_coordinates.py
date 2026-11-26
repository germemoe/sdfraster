"""Training-only depth-normal coordinate selection; no imports, RNG or queries."""

MODES = ("legacy", "ray_jitter")

def validate_depth_normal_coordinate_mode(mode):
    if type(mode) is not str or mode not in MODES:
        raise ValueError("depth_normal_coordinate_mode must be legacy or ray_jitter")
    return mode

def depth_normal_pixel_offsets(mode, ray_jitter):
    """Return the original render offset object, or legacy pixel centers.

    None has the same pixel-center meaning as render(ray_jitter=None).
    No clone, cast, random draw, field query or value-dependent synchronization.
    """
    validate_depth_normal_coordinate_mode(mode)
    return ray_jitter if mode == "ray_jitter" else None
