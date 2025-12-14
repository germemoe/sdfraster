import torch
import math
from torch import nn
from icecream import ic

from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.eval_sh_py import eval_sh
from utils.topo_utils import calculate_circumcenters_torch

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def init_linear(m, gain):
    """Standard Xavier-uniform + zero bias for any Linear layer."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
@torch.jit.script
def pre_calc_cell_values(vertices, indices):
    device = vertices.device
    tets = vertices[indices]
    circumcenter, radius = calculate_circumcenters_torch(tets.double())
    return circumcenter

# @torch.jit.script
def compute_gradient_from_vertex_colors(
    vcolors:        torch.Tensor,   # (T, 4, 3) - (T, V, C)
    element_verts:  torch.Tensor,   # (T, 4, 3) - (T, V, D)
    circumcenters:  torch.Tensor    # (T, 3)    - (T, D)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Returns base (T,C) and gradients (T,C,D)
    """
    Recovers the base color and gradients from vertex colors and geometry.
    Assumes V=4 vertices, D=3 dimensions, C=3 color channels.
    """
    # 1. unit directions from the circum-centre
    # dirs: (T, 4, 3) -> (T, V, D)
    dirs = torch.nn.functional.normalize(
        element_verts - circumcenters[:, None, :], p=2.0, dim=-1, eps=1e-12
    )

    d0, d1, d2, d3 = dirs.unbind(dim=1)
    f0, f1, f2, f3 = vcolors.unbind(dim=1)
    A = torch.stack([d1 - d0, d2 - d0, d3 - d0], dim=1)   # (T, 3, D) = (T, 3, 3)
    
    B = torch.stack([f1 - f0, f2 - f0, f3 - f0], dim=1)   # (T, 3, C)

    solved_G_transposed = torch.linalg.solve(A, B) # Shape (T, D, C)

    gradients_recovered = solved_G_transposed.transpose(1, 2) # Shape (T, C, D) = (T, 3, 3)

    base_recovered = f0 - torch.einsum('tcd,td->tc', gradients_recovered, d0) # Shape (T, C)

    return base_recovered, gradients_recovered

@torch.jit.script
def compute_vertex_colors_from_field(
    element_verts: torch.Tensor,   # (T, 4, 3) - Renamed for clarity (V=4, D=3)
    base:           torch.Tensor,   # (T, 3)    - (T, C)
    gradients:      torch.Tensor,   # (T, 3, 3) - (T, C, D_grad=3)
    circumcenters:  torch.Tensor    # (T, 3)    - (T, D=3)
) -> torch.Tensor: # Returns vertex_colors (T, 4, 3)
    """
    Compute per-vertex colors for each element (e.g., tetrahedron).
    
    For each vertex:
      color = base_for_channel + dot(gradient_for_channel, normalized(vertex - circumcenter))
    """
    offsets = element_verts - circumcenters[:, None, :]
    grad_contrib = torch.einsum('tcd,tvd->tvc', gradients, offsets)
    vertex_colors = base[:, None, :] + grad_contrib 
    
    return vertex_colors

@torch.jit.script
def offset_normalize(rgb, grd, circumcenters, tets):
    grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).min(dim=1, keepdim=True).values
    # grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).mean(dim=1, keepdim=True)#.detach()
    radius = torch.linalg.norm(tets[:, 0] - circumcenters, dim=-1, keepdim=True).reshape(-1, 1, 1)
    normed_grd = safe_div(grd, radius)
    vcolors = compute_vertex_colors_from_field(
        tets.detach(), rgb.reshape(-1, 3), normed_grd.float(), circumcenters.float().detach())

    base_color_v0_raw = vcolors[:, 0]
    return base_color_v0_raw, normed_grd

@torch.jit.script
# [SDF-MODIFICATION] 激活输出函数 - sdf 作为输入参数
def activate_output(camera_center, sdf, rgb, grd, sh, indices, circumcenters, vertices, current_sh_deg:int, max_sh_deg:int):
    # === [DENSITY-RELATED] 原密度代码签名 (已注释) ===
    # def activate_output(camera_center, density, rgb, grd, sh, indices, circumcenters, vertices, current_sh_deg:int, max_sh_deg:int):
    # === [DENSITY-RELATED] 原密度代码结束 ===
    tets = vertices[indices]
    base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
    tet_color_raw = eval_sh(
        tets.mean(dim=1),
        RGB2SH(base_color_v0_raw),
        sh,
        camera_center,
        current_sh_deg).float()
    base_color_v0 = torch.nn.functional.softplus(tet_color_raw.reshape(-1, 3, 1), beta=10)
    # [SDF-MODIFICATION] sdf 作为特征的第一个维度 (dim=0)
    # TODO: 后续渲染代码需要将SDF转换为alpha (使用NeuS公式)
    features = torch.cat([sdf, base_color_v0.reshape(-1, 3), normed_grd.reshape(-1, 3)], dim=1)
    # === [DENSITY-RELATED] 原密度代码 (已注释) ===
    # features = torch.cat([density, base_color_v0.reshape(-1, 3), normed_grd.reshape(-1, 3)], dim=1)
    # === [DENSITY-RELATED] 原密度代码结束 ===
    return features.float()


# ============================================================================
# Eikonal Loss 和 Curvature Loss（参考 Neuralangelo）
# ============================================================================

def eikonal_loss(gradients: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Eikonal Loss: 约束 SDF 梯度的范数为 1

    Args:
        gradients: (N, 3) SDF 的梯度向量
        mask: (N,) 可选的掩码，只在 mask=True 的位置计算 loss

    Returns:
        scalar loss
    """
    # ||∇SDF|| - 1 的平方
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # (N,)
    gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        return (gradient_error * mask.float()).sum() / (mask.float().sum() + 1e-8)
    else:
        return gradient_error.mean()


def curvature_loss(hessian_diag: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Curvature Loss: 最小化 SDF 的拉普拉斯量（促进光滑表面）

    Args:
        hessian_diag: (N, 3) Hessian 对角元素 [∂²f/∂x², ∂²f/∂y², ∂²f/∂z²]
        mask: (N,) 可选的掩码，只在 mask=True 的位置计算 loss

    Returns:
        scalar loss
    """
    # Laplacian = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    laplacian = hessian_diag.sum(dim=-1).abs()  # (N,)
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    if mask is not None:
        return (laplacian * mask.float()).sum() / (mask.float().sum() + 1e-8)
    else:
        return laplacian.mean()