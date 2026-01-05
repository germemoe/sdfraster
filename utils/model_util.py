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


def sample_eikonal_points(model, num_samples: int = 50000, band: float = 0.1, use_tet_sampling: bool = True):
    """
    采样 Eikonal loss 计算点

    策略：
    1. 找满足条件的候选点：|SDF| < band 的内部顶点（band=None 时使用所有内部顶点）
    2. 如果候选点数量 >= num_samples：用 topk 取 |SDF| 最小的 num_samples 个点
    3. 如果候选点数量 < num_samples 且 use_tet_sampling=True：
       在候选点邻域球体内采样补足到目标数量

    Args:
        model: Model 实例
        num_samples: 目标总采样点数
        band: SDF band 阈值（世界坐标），None 表示使用所有内部顶点
        use_tet_sampling: 当候选点不足时，是否在邻域内采样补足

    Returns:
        all_points: (M, 3) 采样点坐标
        vertex_indices: (num_band,) band 内顶点的全局索引（用于查找边）
        num_band: band 内顶点数（原始候选点数）
        num_tet_samples: 邻域内额外采样点数
    """
    device = model.device

    # 1. 只在内部顶点上操作（排除 ext_vertices）
    num_internal = model.num_int_verts
    internal_verts = model.vertices[:num_internal]  # (num_internal, 3)

    with torch.no_grad():
        vert_sdf = model.query_vert_sdf(internal_verts)  # (num_internal,)

    # 2. 筛选候选点
    if band is not None:
        candidate_mask = vert_sdf.abs() < band
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    else:
        # 所有内部顶点都是候选
        candidate_indices = torch.arange(num_internal, device=device)

    num_candidates = candidate_indices.shape[0]

    # 3. 根据候选点数量决定策略
    if num_candidates >= num_samples:
        # 候选点足够：用 topk 取 |SDF| 最小的 num_samples 个
        candidate_sdf_abs = vert_sdf[candidate_indices].abs()
        _, topk_indices = candidate_sdf_abs.topk(num_samples, largest=False)
        selected_indices = candidate_indices[topk_indices]
        band_points = internal_verts[selected_indices]
        return band_points, selected_indices, num_samples, 0

    # 4. 候选点不足
    band_points = internal_verts[candidate_indices]
    num_band = num_candidates

    if not use_tet_sampling or num_band == 0:
        return band_points, candidate_indices, num_band, 0

    # 5. 在候选点邻域球体内随机采样补足
    num_need = num_samples - num_band

    # 计算步长：使用 eps（数值梯度步长）作为邻域半径
    if hasattr(model.backbone, 'get_normal_epsilon'):
        eps = model.backbone.get_normal_epsilon() * float(model.scene_scaling)
    else:
        finest_res = model.backbone.base_resolution * (model.backbone.per_level_scale ** (model.backbone.L - 1))
        eps = float(model.scene_scaling) / finest_res

    # 从现有候选点中随机选取基点（允许重复）
    base_indices = torch.randint(0, num_band, (num_need,), device=device)
    base_points = band_points[base_indices]

    # 球体内均匀采样：方向随机 + 半径 ~ U(0, 3*eps)
    random_dirs = torch.randn(num_need, 3, device=device)
    random_dirs = random_dirs / (random_dirs.norm(dim=-1, keepdim=True) + 1e-8)
    random_radius = torch.rand(num_need, 1, device=device) * (3 * eps)  # U(0, 3*eps)
    sampled_points = base_points + random_dirs * random_radius

    # 6. 合并
    all_points = torch.cat([band_points, sampled_points], dim=0)
    return all_points, candidate_indices, num_band, sampled_points.shape[0]


def extract_edges_from_tets(indices: torch.Tensor) -> torch.Tensor:
    """
    从四面体索引中提取所有唯一的边

    Args:
        indices: (T, 4) 四面体顶点索引

    Returns:
        edges: (E, 2) 唯一边的顶点索引对，已排序（小索引在前）
    """
    # 每个四面体有 6 条边: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_pairs = torch.tensor([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
    ], device=indices.device)

    # 提取所有边 (T*6, 2)
    all_edges = indices[:, edge_pairs].reshape(-1, 2)

    # 排序每条边（小索引在前），便于去重
    all_edges = all_edges.sort(dim=1).values

    # 去重
    unique_edges = torch.unique(all_edges, dim=0)

    return unique_edges


def edge_sdf_loss(
    model,
    vert_sdf: torch.Tensor = None,
    sdf_threshold: float = 0.1,
    edge_length_threshold: float = None,
    target: float = None
):
    """
    Edge SDF Loss：约束边两端顶点的 SDF 变化率

    逻辑：
    - diff / edge_length 归一化，消除边长对 SDF 差异的影响
    - 只考虑表面附近的边（至少一端 |SDF| < sdf_threshold）
    - 排除包含外部点的边

    Args:
        model: Model 实例
        vert_sdf: 预计算的顶点 SDF，None 则现场计算（只计算内部点）
        sdf_threshold: SDF 阈值，只考虑表面附近的边
        edge_length_threshold: 边长阈值，只考虑短边（可选）
        target: 目标值，None 则最小化 diff/length（平滑），
                设为 1.0 则约束 ((diff/length) - 1)^2（Eikonal 风格）

    Returns:
        loss: scalar
        num_edges: 参与计算的边数
    """
    device = model.device
    num_internal = model.num_int_verts

    # 1. 获取内部边（使用缓存）
    edges = model.get_internal_edges()  # (E, 2)，已排除外部点

    if edges.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 2. 计算边长（detach，不让梯度流向顶点位置）
    vertices = model.vertices
    v0, v1 = vertices[edges[:, 0]], vertices[edges[:, 1]]
    edge_length = (v0 - v1).norm(dim=-1).detach()  # (E,)

    # 3. 获取顶点 SDF（只查询内部点，需要梯度以便反向传播）
    if vert_sdf is None:
        internal_verts = model.vertices[:num_internal]
        vert_sdf = model.query_vert_sdf(internal_verts)  # (num_internal,)

    sdf_a, sdf_b = vert_sdf[edges[:, 0]], vert_sdf[edges[:, 1]]
    diff = (sdf_a - sdf_b).abs()

    # 4. 筛选：表面附近的边（至少一端接近表面）
    near_surface = (sdf_a.abs() < sdf_threshold) | (sdf_b.abs() < sdf_threshold)

    # 5. 筛选：短边（可选）
    if edge_length_threshold is not None:
        short_edge = edge_length < edge_length_threshold
        mask = near_surface & short_edge
    else:
        mask = near_surface

    num_edges = mask.sum().item()
    if num_edges == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 6. 计算 loss
    ratio = diff[mask] / (edge_length[mask] + 1e-8)

    if target is not None:
        # Eikonal 风格：约束 ratio 接近 target
        loss = ((ratio - target) ** 2).mean()
    else:
        # 平滑风格：最小化 ratio
        loss = ratio.mean()

    return loss, num_edges


def gradient_tv_loss(
    model,
    vert_sdf: torch.Tensor = None,
    vert_gradients: torch.Tensor = None,
    sdf_threshold: float = 0.1,
    edge_length_threshold: float = None,
    normalize_by_edge_length: bool = True
):
    """
    Gradient TV Loss：约束边两端顶点的 SDF 梯度变化

    促进表面法向量的平滑性，减少噪声

    逻辑：
    - 计算边两端顶点梯度的 L2 距离
    - 只考虑表面附近的边（至少一端 |SDF| < sdf_threshold）
    - 排除包含外部点的边

    Args:
        model: Model 实例
        vert_sdf: 预计算的顶点 SDF，None 则现场计算（只计算内部点）
        vert_gradients: 预计算的顶点梯度 (N, 3)，None 则现场计算
        sdf_threshold: SDF 阈值，只考虑表面附近的边
        edge_length_threshold: 边长阈值，只考虑短边（可选）
        normalize_by_edge_length: 是否用边长归一化梯度差异

    Returns:
        loss: scalar
        num_edges: 参与计算的边数
    """
    device = model.device
    num_internal = model.num_int_verts

    # 1. 获取内部边（使用缓存）
    edges = model.get_internal_edges()  # (E, 2)，已排除外部点

    if edges.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 2. 计算边长（detach，不让梯度流向顶点位置）
    vertices = model.vertices
    v0, v1 = vertices[edges[:, 0]], vertices[edges[:, 1]]
    edge_length = (v0 - v1).norm(dim=-1).detach()  # (E,)

    # 3. 获取顶点 SDF 和梯度
    internal_verts = model.vertices[:num_internal]

    if vert_sdf is None or vert_gradients is None:
        # 计算 SDF 和梯度
        sdf, gradients, _ = model.compute_sdf_gradients(internal_verts, mode="numerical", taps=6)
        if vert_sdf is None:
            vert_sdf = sdf
        if vert_gradients is None:
            vert_gradients = gradients

    # 4. 获取边两端的 SDF 和梯度
    sdf_a, sdf_b = vert_sdf[edges[:, 0]], vert_sdf[edges[:, 1]]
    grad_a, grad_b = vert_gradients[edges[:, 0]], vert_gradients[edges[:, 1]]  # (E, 3)

    # 计算梯度 L2 距离
    grad_diff = (grad_a - grad_b).norm(dim=-1)  # (E,)

    # 5. 筛选：表面附近的边（至少一端接近表面）
    near_surface = (sdf_a.abs() < sdf_threshold) | (sdf_b.abs() < sdf_threshold)

    # 6. 筛选：短边（可选）
    if edge_length_threshold is not None:
        short_edge = edge_length < edge_length_threshold
        mask = near_surface & short_edge
    else:
        mask = near_surface

    num_edges = mask.sum().item()
    if num_edges == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 7. 计算 loss
    if normalize_by_edge_length:
        # 用边长归一化，使梯度变化率与边长无关
        loss = (grad_diff[mask] / (edge_length[mask] + 1e-8)).mean()
    else:
        # 直接使用梯度差异
        loss = grad_diff[mask].mean()

    return loss, num_edges


def gradient_tv_loss_surface_edges(
    model,
    normalize_by_edge_length: bool = True,
    gradient_mode: str = "analytical",
    gradient_taps: int = 4,
    gradient_eps: float = 1e-3
):
    """
    Gradient TV Loss（穿过表面的边版本）

    只计算穿过表面的边（两端点 SDF 异号）的梯度变化。

    Args:
        model: Model 实例
        normalize_by_edge_length: 是否用边长归一化梯度差异
        gradient_mode: 梯度计算模式 ("analytical" 或 "numerical")
        gradient_taps: 数值梯度采样点数
        gradient_eps: 数值梯度步长

    Returns:
        loss: scalar
        num_edges: 参与计算的边数
    """
    device = model.device
    num_internal = model.num_int_verts

    # 1. 获取内部边
    edges = model.get_internal_edges()  # (E, 2)
    if edges.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 2. 计算内部顶点的 SDF
    internal_verts = model.vertices[:num_internal]
    with torch.no_grad():
        vert_sdf = model.query_vert_sdf(internal_verts)  # (num_internal,)

    # 3. 筛选穿过表面的边（两端点 SDF 异号）
    sdf_a = vert_sdf[edges[:, 0]]
    sdf_b = vert_sdf[edges[:, 1]]
    surface_mask = (sdf_a * sdf_b) < 0  # 异号

    surface_edges = edges[surface_mask]  # (E', 2)
    num_edges = surface_edges.shape[0]

    if num_edges == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 4. 获取这些边的端点（需要梯度）
    edge_vert_indices = torch.unique(surface_edges.flatten())  # 去重
    edge_verts = model.vertices[edge_vert_indices]  # (K, 3)

    # 5. 计算这些顶点的 SDF 梯度
    _, gradients, _ = model.compute_sdf_gradients(
        points=edge_verts,
        mode=gradient_mode,
        taps=gradient_taps,
        eps=gradient_eps
    )  # (K, 3)

    # 6. 构建索引映射
    max_idx = num_internal
    global_to_local = torch.full((max_idx,), -1, dtype=torch.long, device=device)
    global_to_local[edge_vert_indices] = torch.arange(edge_vert_indices.shape[0], device=device)

    local_a = global_to_local[surface_edges[:, 0]]
    local_b = global_to_local[surface_edges[:, 1]]

    # 7. 计算边长
    v0, v1 = model.vertices[surface_edges[:, 0]], model.vertices[surface_edges[:, 1]]
    edge_length = (v0 - v1).norm(dim=-1).detach()

    # 8. 获取梯度并计算差异
    grad_a = gradients[local_a]
    grad_b = gradients[local_b]
    grad_diff = (grad_a - grad_b).norm(dim=-1)

    # 9. 计算 loss
    if normalize_by_edge_length:
        loss = (grad_diff / (edge_length + 1e-8)).mean()
    else:
        loss = grad_diff.mean()

    return loss, num_edges


def gradient_tv_loss_from_sampled(
    model,
    vertex_indices: torch.Tensor,
    vert_gradients: torch.Tensor,
    normalize_by_edge_length: bool = True
):
    """
    Gradient TV Loss（复用 Eikonal 采样点版本）

    只计算 Eikonal 采样点之间的边的梯度变化，避免重复计算梯度。

    逻辑：
    - 使用 sample_eikonal_points 返回的 vertex_indices（band 内顶点的全局索引）
    - 只考虑两端点都在 vertex_indices 中的边
    - 使用 Eikonal 计算时已经得到的梯度，不重新计算

    Args:
        model: Model 实例
        vertex_indices: (K,) Eikonal 采样的顶点全局索引（来自 sample_eikonal_points）
        vert_gradients: (K, 3) 这些顶点的 SDF 梯度（已在 Eikonal 计算时得到）
        normalize_by_edge_length: 是否用边长归一化梯度差异

    Returns:
        loss: scalar
        num_edges: 参与计算的边数
    """
    device = model.device

    if vertex_indices.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 1. 获取内部边（使用缓存）
    edges = model.get_internal_edges()  # (E, 2)，已排除外部点

    if edges.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 2. 构建全局索引到局部索引的映射
    # vertex_indices[local_idx] = global_idx
    # 我们需要 global_to_local: global_idx -> local_idx
    num_sampled = vertex_indices.shape[0]
    max_vert_idx = model.vertices.shape[0]

    # 使用 scatter 构建映射（-1 表示不在采样集中）
    global_to_local = torch.full((max_vert_idx,), -1, dtype=torch.long, device=device)
    local_indices = torch.arange(num_sampled, device=device)
    global_to_local[vertex_indices] = local_indices

    # 3. 筛选两端点都在 vertex_indices 中的边
    local_a = global_to_local[edges[:, 0]]  # (E,)
    local_b = global_to_local[edges[:, 1]]  # (E,)
    valid_mask = (local_a >= 0) & (local_b >= 0)

    valid_edges = edges[valid_mask]  # (E', 2)
    valid_local_a = local_a[valid_mask]  # (E',)
    valid_local_b = local_b[valid_mask]  # (E',)

    num_edges = valid_edges.shape[0]
    if num_edges == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    # 4. 计算边长（detach，不让梯度流向顶点位置）
    vertices = model.vertices
    v0, v1 = vertices[valid_edges[:, 0]], vertices[valid_edges[:, 1]]
    edge_length = (v0 - v1).norm(dim=-1).detach()  # (E',)

    # 5. 获取边两端的梯度（使用局部索引从 vert_gradients 中获取）
    grad_a = vert_gradients[valid_local_a]  # (E', 3)
    grad_b = vert_gradients[valid_local_b]  # (E', 3)

    # 6. 计算梯度 L2 距离
    grad_diff = (grad_a - grad_b).norm(dim=-1)  # (E',)

    # 7. 计算 loss
    if normalize_by_edge_length:
        loss = (grad_diff / (edge_length + 1e-8)).mean()
    else:
        loss = grad_diff.mean()

    return loss, num_edges