"""Stateless Eikonal candidate selection and CPU-safe configuration metadata.

Uniform sampling uses a device-local Generator bound to (training seed, step).
It never saves mutable RNG state or advances the process-global RNG streams.
"""
import hashlib

EIKONAL_SAMPLING_MODES = ("nearest_abs_sdf", "uniform")
UNIFORM_SAMPLING_VERSION = "torch_randperm_sha256_seed_step_v1"


def validate_eikonal_sampling_mode(mode):
    if not isinstance(mode, str) or mode not in EIKONAL_SAMPLING_MODES:
        raise ValueError("eikonal_sampling_mode must be nearest_abs_sdf or uniform")
    return mode


def _integer(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def uniform_step_seed(seed, step):
    """Stable, versioned seed mixing; independent of Python hash/global RNG."""
    seed = _integer(seed, "training seed")
    step = _integer(step, "sampling step", nonnegative=True)
    payload = f"sdfraster.eikonal_uniform.v1\0{seed}\0{step}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def eikonal_sampling_metadata(mode, seed, step=None):
    mode = validate_eikonal_sampling_mode(mode)
    seed = _integer(seed, "training seed")
    if step is not None:
        step = _integer(step, "sampling step", nonnegative=True)
    uniform = mode == "uniform"
    return {
        "mode": mode,
        "training_seed": seed,
        "bound_step": step,
        "step_semantics": "zero_based_iteration_of_this_run; --ckpt remains weights_only",
        "candidate_pool": "current_internal_vertices_filtered_by_configured_abs_sdf_band",
        "rng_policy": "device_local_generator_bound_to_seed_and_step" if uniform
                      else "legacy_topk; optional_neighborhood_uses_legacy_global_rng",
        "algorithm_version": UNIFORM_SAMPLING_VERSION if uniform else "legacy_abs_sdf_topk",
        "derived_seed": uniform_step_seed(seed, step) if uniform and step is not None else None,
        "cross_device_version_bitwise_guarantee": False,
    }


def validate_eikonal_sampling_configuration(config):
    """Validate before dataset setup and in plan ready checks; no torch import."""
    def get(name, default):
        return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)
    mode = validate_eikonal_sampling_mode(get("eikonal_sampling_mode", "nearest_abs_sdf"))
    seed = get("seed", 52)
    if mode == "uniform":
        count = _integer(get("num_eikonal_samples", 500000), "num_eikonal_samples")
        if count <= 0:
            raise ValueError("uniform Eikonal sampling requires positive num_eikonal_samples")
        if get("eikonal_tet_sampling", False) is not False:
            raise ValueError("uniform Eikonal sampling requires eikonal_tet_sampling=False; "
                             "it samples existing candidates without replacement and never adds jitter")
    return eikonal_sampling_metadata(mode, seed)


def uniform_candidate_indices(candidate_indices, num_samples, seed, step):
    """Select min(K, N) existing candidates, without replacement or SDF queries.

    A full/undersized pool retains its order. With N > K, randperm is evaluated
    on the candidates' device with a fresh local, deterministically seeded RNG.
    """
    import torch
    if candidate_indices.ndim != 1 or candidate_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("candidate_indices must be a one-dimensional integer tensor")
    num_samples = _integer(num_samples, "num_eikonal_samples")
    if num_samples <= 0:
        raise ValueError("uniform Eikonal sampling requires positive num_eikonal_samples")
    derived_seed = uniform_step_seed(seed, step)
    if candidate_indices.numel() <= num_samples:
        return candidate_indices
    generator = torch.Generator(device=candidate_indices.device)
    generator.manual_seed(derived_seed)
    order = torch.randperm(candidate_indices.numel(), device=candidate_indices.device,
                           generator=generator)[:num_samples]
    return candidate_indices[order]
