"""Geometry diagnostics for raw-gradient normal weighting.

These correlations describe per-tetrahedron gradient magnitudes. They do not
measure a tetrahedron's final pixel influence, which also depends on opacity,
visibility and transmittance.
"""
import torch
from utils.topo_utils import calculate_circumcenters_torch


def _sample_tetrahedron_rows(count, max_tets, device):
    # Float32 linspace can round count - 1 up to count on large grids.
    sample_count = min(count, max_tets)
    rows = torch.arange(sample_count, device=device, dtype=torch.int64)
    return rows * (count - 1) // max(sample_count - 1, 1)


def _correlation(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= torch.finfo(x.dtype).eps:
        return None
    return float((x.dot(y) / denominator).cpu())


@torch.no_grad()
def normal_geometry_diagnostics(model, max_tets=10000):
    count = model.indices.shape[0]
    if count == 0:
        return {"status": "empty_grid", "sampled_tetrahedra": 0}
    selected = _sample_tetrahedron_rows(count, max_tets, model.indices.device)
    indices = model.indices[selected].long()
    vertices = model.vertices[indices]
    edges = (vertices[:, 1:] - vertices[:, :1]).double()
    determinant = torch.linalg.det(edges)
    magnitude = edges.norm(dim=-1).prod(dim=-1)
    valid = (determinant.abs() > 32 * torch.finfo(edges.dtype).eps *
             magnitude.clamp_min(torch.finfo(edges.dtype).tiny))
    valid &= torch.isfinite(edges).flatten(start_dim=1).all(dim=1)
    if not valid.any():
        return {"status": "no_nondegenerate_samples",
                "sampled_tetrahedra": int(selected.numel())}
    vertices = vertices[valid]
    edges = edges[valid]
    flat_vertices = vertices.reshape(-1, 3)
    sdf = model.query_vert_sdf(flat_vertices).double()
    if getattr(model, "ext_sdf_fixed", False):
        external = indices[valid].reshape(-1) >= model.num_int_verts
        sdf[external] = ((flat_vertices[external] - model.center) /
                         model.scene_scaling).norm(dim=-1).double()
    sdf = sdf.reshape(-1, 4)
    gradient = torch.linalg.solve(edges, (sdf[:, 1:] - sdf[:, :1]).unsqueeze(-1)).squeeze(-1)
    gradient_norm = gradient.norm(dim=-1)
    _, radius = calculate_circumcenters_torch(vertices.double())
    radius = radius.reshape(-1)
    volume = determinant[valid].abs() / 6
    finite = torch.isfinite(gradient_norm) & torch.isfinite(radius) & (radius > 0)
    gradient_norm, radius, volume = gradient_norm[finite], radius[finite], volume[finite]
    if gradient_norm.numel() == 0:
        return {"status": "no_finite_samples",
                "sampled_tetrahedra": int(selected.numel())}
    tiny = torch.finfo(gradient_norm.dtype).tiny
    quantiles = torch.tensor([0., .25, .5, .75, 1.], device=radius.device, dtype=radius.dtype)
    return {
        "status": "ok",
        "sampled_tetrahedra": int(selected.numel()),
        "valid_tetrahedra": int(gradient_norm.numel()),
        "gradient_norm_quantiles": torch.quantile(gradient_norm, quantiles).cpu().tolist(),
        "circumradius_quantiles": torch.quantile(radius, quantiles).cpu().tolist(),
        "volume_quantiles": torch.quantile(volume, quantiles).cpu().tolist(),
        "corr_log_radius_log_gradient_norm": _correlation(
            radius.clamp_min(tiny).log(), gradient_norm.clamp_min(tiny).log()),
        "corr_log_volume_log_gradient_norm": _correlation(
            volume.clamp_min(tiny).log(), gradient_norm.clamp_min(tiny).log()),
        "scope": "Geometry correlation; not alpha-composited pixel contribution",
    }
