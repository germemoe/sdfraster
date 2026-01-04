import gc
import cv2
import torch
import imageio
from utils import safe_math
from typing import NamedTuple, List
from delaunay_rasterization.internal.render_err import render_err
from icecream import ic


@torch.no_grad()
def determine_cull_mask(
    sampled_cameras: List["Camera"],
    model,
    # glo_list,
    args,
    device: torch.device,
):
    """Accumulate densification statistics for one iteration."""
    n_tets = model.indices.shape[0]
    peak_contrib = torch.zeros((n_tets), device=device)

    for cam in sampled_cameras:
        target = cam.original_image.cuda()

        image_votes, extras = render_err(
            target, cam, model,
            scene_scaling=model.scene_scaling,
            tile_size=args.tile_size,
            # glo=glo_list(torch.LongTensor([cam.uid]).to(device))
        )

        tc = extras["tet_count"][..., 0]
        max_T = extras["tet_count"][..., 1].float() / 65535
        # peak_contrib = torch.maximum(image_T / tc.clip(min=1), peak_contrib)
        peak_contrib = torch.maximum(max_T, peak_contrib)

    mask = ((peak_contrib > args.contrib_threshold))
    return mask

def get_approx_ray_intersections(split_rays_data, epsilon=1e-7):
    """
    Calculates the approximate intersection point for pairs of line segments.

    The intersection is defined as the midpoint of the shortest segment
    connecting the two input line segments.

    Args:
        split_rays_data (torch.Tensor): Tensor of shape (N, 2, 6).
            - N: Number of segment pairs.
            - 2: Represents the two segments in a pair.
            - 6: Contains [Ax, Ay, Az, Bx, By, Bz] for each segment,
                 where A and B are the segment endpoints.
                 Based on current Python code:
                 A = average_P_exit, B = average_P_entry
        epsilon (float): Small value to handle parallel lines and avoid
                         division by zero if a segment has zero length.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the approximate
                      "intersection" points (midpoints of closest approach).
    """
    # Segment 1 endpoints
    p1_a = split_rays_data[:, 0, 0:3]  # Endpoint A of first segments (N, 3)
    p1_b = split_rays_data[:, 0, 3:6]  # Endpoint B of first segments (N, 3)
    # Segment 2 endpoints
    p2_a = split_rays_data[:, 1, 0:3]  # Endpoint A of second segments (N, 3)
    p2_b = split_rays_data[:, 1, 3:6]  # Endpoint B of second segments (N, 3)

    # Define segment origins and direction vectors
    # Segment S1: o1 + s * d1, for s in [0, 1]
    # Segment S2: o2 + t * d2, for t in [0, 1]
    o1 = p1_a
    d1 = p1_b - p1_a  # Direction vector for segment 1 (from A to B)
    o2 = p2_a
    d2 = p2_b - p2_a  # Direction vector for segment 2 (from A to B)

    # Calculate terms for finding closest points on the infinite lines
    # containing the segments (based on standard formulas, e.g., Christer Ericson's "Real-Time Collision Detection")
    v_o = o1 - o2 # Vector from origin of line 2 to origin of line 1

    a = torch.sum(d1 * d1, dim=1)  # Squared length of d1
    b = torch.sum(d1 * d2, dim=1)  # Dot product of d1 and d2
    c = torch.sum(d2 * d2, dim=1)  # Squared length of d2
    d = torch.sum(d1 * v_o, dim=1) # d1 dot (o1 - o2)
    e = torch.sum(d2 * v_o, dim=1) # d2 dot (o1 - o2)

    denom = a * c - b * b
    
    # Parameters for closest points on the *infinite lines*
    # s_line = (b*e - c*d) / denom
    # t_line = (a*e - b*d) / denom (this t_line corresponds to -t in some formulations, careful with sign)
    # The t_line should be for the parameterization o2 + t*d2.
    # If P1 = o1 + s*d1 and P2 = o2 + t*d2, and we minimize ||P1-P2||^2,
    # by setting derivatives w.r.t s and t to 0, we get:
    # s * (d1.d1) - t * (d1.d2) = -d1.(o1-o2) = d1.v_o = d
    # s * (d1.d2) - t * (d2.d2) = -d2.(o1-o2) = d2.v_o = e
    # Solving this system:
    # s_line = (d*c - e*b) / denom
    # t_line = (d*b - e*a) / denom -> this results in parameter for -d2 if system set up for P1-P2
    # Or, more directly for t_line for P2 = o2 + t*d2: t_line = (b*d - a*e) / denom
    
    s_line_num = (b * e) - (c * d)
    t_line_num = (a * e) - (b * d) # This corresponds to t_c = (a*e - b*d)/denom from previous thoughts for P(t) = O2 + tD2

    # Handle near-zero denominator (lines are parallel or one segment is a point)
    # We compute with a safe denominator, then clamp. Clamping is key for segments.
    denom_safe = torch.where(denom.abs() < epsilon, torch.ones_like(denom), denom)
    
    s_line = s_line_num / denom_safe
    t_line = t_line_num / denom_safe # Note: This t_line is for the parameter of d2 (from o2)

    # Clamp parameters to [0, 1] to stay within the segments
    bad_intersect = (s_line < 0) | (t_line < 0) | (s_line > 1) | (t_line > 1)
    s_seg = torch.clamp(s_line, 0.0, 1.0)
    t_seg = torch.clamp(t_line, 0.0, 1.0)

    # Points of closest approach on the segments
    pc1 = o1 + s_seg.unsqueeze(1) * d1
    pc2 = o2 + t_seg.unsqueeze(1) * d2
    
    p_int = (pc1 + pc2) / 2.0
                        
    return p_int, bad_intersect

class RenderStats(NamedTuple):
    within_var_rays: torch.Tensor         # (T, 2, 6)
    total_var_moments: torch.Tensor     # (T, 3)
    tet_moments: torch.Tensor           # (T, 4)
    tet_view_count: torch.Tensor             # (T,)
    peak_contrib: torch.Tensor              # (T,)
    top_ssim: torch.Tensor
    top_size: torch.Tensor


@torch.no_grad()
def collect_render_stats(
    sampled_cameras: List["Camera"],
    model,
    args,
    device: torch.device,
):
    """Accumulate densification statistics for one iteration."""
    n_tets = model.indices.shape[0]

    # Pre-allocate accumulators ------------------------------------------------
    tet_moments = torch.zeros((n_tets, 4), device=device)
    tet_view_count = torch.zeros((n_tets,), device=device)

    top_ssim = torch.zeros((n_tets, 2), device=device)
    top_size = torch.zeros((n_tets, 2), device=device)
    peak_contrib = torch.zeros((n_tets), device=device)
    within_var_rays = torch.zeros((n_tets, 2, 6), device=device)
    total_var_moments = torch.zeros((n_tets, 3), device=device)
    top_moments = torch.zeros((n_tets, 2, 4), device=device)

    # Main per-camera loop -----------------------------------------------------
    for cam in sampled_cameras:
        target = cam.original_image.cuda()

        image_votes, extras = render_err( target, cam, model, tile_size=args.tile_size)

        tc = extras["tet_count"][..., 0]
        max_T = extras["tet_count"][..., 1].float() / 65535
        peak_contrib = torch.maximum(max_T, peak_contrib)
        
        # --- Create a single mask for valid updates ---
        # Mask for tets that have a reasonable number of samples in the current view
        update_mask = (tc >= args.min_tet_count) & (tc < 8000)

        # --- Moments (s0: sum of T, s1: sum of err, s2: sum of err^2)
        image_T, image_err, image_err2 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        _, _, image_ssim = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        N = tc
        image_ssim[~update_mask] = 0

        # -------- Within-Image Variance (Top-2 per tet) -----------------------
        within_var_mu = safe_math.safe_div(image_err, N)
        within_var_std = (safe_math.safe_div(image_err2, N) - within_var_mu**2).clip(min=0)
        within_var_std[N < 10] = 0
        within_var_std[~update_mask] = 0 # Use the unified mask

        # ray buffer: (enter | exit) → (N, 6)
        w = image_votes[:, 12:13]
        seg_exit = safe_math.safe_div(image_votes[:, 9:12], w)
        seg_enter = safe_math.safe_div(image_votes[:, 6:9], w)

        image_ssim = image_ssim / tc.clip(min=1)

        # keep top-2 candidates per tet across all views
        top_ssim, idx_sorted = torch.cat([top_ssim[:, :2], image_ssim.reshape(-1, 1)], dim=1).sort(1, descending=True)

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)
        moments = torch.cat([
            image_votes[:, 13:16],
            w.reshape(-1, 1)
        ], dim=1)
        moments_3 = torch.cat([top_moments, moments.reshape(-1, 1, 4)], dim=1)
        top_moments = torch.gather(
            moments_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 4)
        )

        rays = torch.cat([seg_enter, seg_exit], dim=1)
        rays_3 = torch.cat([within_var_rays, rays[:, None]], dim=1)
        within_var_rays = torch.gather(
            rays_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 6)
        )

        # -------- Total Variance (accumulated across images) ------------------
        total_var_moments[update_mask, 0] += image_T[update_mask]
        total_var_moments[update_mask, 1] += image_err[update_mask]
        total_var_moments[update_mask, 2] += image_err2[update_mask]

        # -------- Between-Image Variance (accumulated across images) ----------
        # We compute the variance of the mean error across different views
        mean_err_per_view = within_var_mu
        mean_err_per_view[N < 10] = 0

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)

        tet_view_count[update_mask] += 1 # Count views per tet

    # done
    return RenderStats(
        within_var_rays = within_var_rays,
        total_var_moments = total_var_moments,
        tet_moments = tet_moments,
        tet_view_count = tet_view_count,
        top_ssim = top_ssim[:, :2],
        top_size = top_size[:, :2],
        peak_contrib = peak_contrib # used for determining what to clone
    )

@torch.no_grad()
def apply_densification(
    stats: RenderStats,
    model,
    tet_optim,
    args,
    iteration: int,
    device: torch.device,
    sample_cam,
    sample_image,
    target_addition
):
    """Turns accumulated statistics into actual vertex cloning / splitting."""
    # ---------- Calculate scores from variances ------------------------------
    # 1. Total Variance Score (for growing)
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu**2).clip(min=0)
    total_var_std[s0_t < 1] = 0

    N_b = stats.tet_view_count # Num views
    within_var = (stats.top_ssim).sum(dim=1)
    total_var = s0_t * total_var_std
    total_var[(N_b < 2) | (s0_t < 1)] = 0

    # --- Masking and target calculation --------------------------------------
    mask_alive = stats.peak_contrib > args.clone_min_contrib
    total_var[stats.peak_contrib < args.clone_min_contrib] = 0
    within_var[stats.peak_contrib < args.split_min_contrib] = 0

    keep_verts = torch.zeros((model.vertices.shape[0]), dtype=torch.int, device=device)
    indices = model.indices.long()
    reduce_type = "sum"
    mask_alive_i = mask_alive.int()
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 0], src=mask_alive_i, reduce=reduce_type)
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 1], src=mask_alive_i, reduce=reduce_type)
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 2], src=mask_alive_i, reduce=reduce_type)
    keep_verts.scatter_reduce_(dim=0, index=indices[..., 3], src=mask_alive_i, reduce=reduce_type)

    target_addition = int(min(target_addition, stats.tet_view_count.shape[0]))
    if target_addition < 0:
        return


    total_mask = torch.zeros_like(total_var, dtype=torch.bool)
    within_mask = torch.zeros_like(total_mask)

    # if target_total > 0:
    #     top_total = torch.topk(temp_total_score, target_total).indices
    #     ic(temp_total_score[top_total].min())
    #     total_mask[top_total] = temp_total_score[top_total] > 0
    #     temp_within_score[total_mask] = 0

    # if target_within > 0:
    #     top_within = torch.topk(temp_within_score, target_within).indices
    #     ic(temp_within_score[top_within].min())
    #     within_mask[top_within] = temp_within_score[top_within] > 0
    
    # clone_mask = within_mask | total_mask

    if args.output_path is not None:

        # f = mask_alive.float().unsqueeze(1).expand(-1, 4).clone()
        # color = torch.rand_like(f[:, :3])
        # # color = rgb + 0.5#torch.rand_like(f[:, :3])
        # f[:, :3] = color
        # f[:, 3] *= 2.0    # alpha
        # imageio.imwrite(args.output_path / f"alive_mask{iteration}.png",
        #                 render_debug(f, model, sample_cam, 10, tile_size=args.tile_size))
        # f = clone_mask.float().unsqueeze(1).expand(-1, 4).clone()
        # f[:, :3] = color
        # f[:, 3] *= 2.0    # alpha
        # imageio.imwrite(args.output_path / f"densify{iteration}.png",
        #                 render_debug(f, model, sample_cam, 10, tile_size=args.tile_size))
        # imageio.imwrite(args.output_path / f"total_var{iteration}.png",
        #                 render_debug(total_var[:, None],
        #                              model, sample_cam, tile_size=args.tile_size))
        # imageio.imwrite(args.output_path / f"within_var{iteration}.png",
        #                 render_debug(within_var[:, None],
        #                              model, sample_cam, tile_size=args.tile_size))
        imageio.imwrite(args.output_path / f"im{iteration}.png",
                        cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

    within_mask = (within_var > args.within_thresh)
    total_mask = (total_var > args.total_thresh)
    error_driven_mask = within_mask | total_mask
    n_error_split = error_driven_mask.sum().item()

    # [SDF-DENSIFY] 基于 SDF 的分裂策略
    # 数量 = 当前顶点数 × ratio
    # 选择跨越表面（异号SDF）且外接圆最大的 TopK 四面体
    # 分裂点 = 四面体重心
    sdf_driven_mask = torch.zeros_like(error_driven_mask, dtype=torch.bool)
    sdf_split_points = None  # 单独存储 SDF 分裂的点位置

    if (hasattr(args, 'sdf_densify_enabled') and args.sdf_densify_enabled
        and iteration >= getattr(args, 'sdf_densify_start', 5000)
        and iteration <= getattr(args, 'sdf_densify_end', 15000)):

        # SDF 分裂数量 = 当前顶点数 × ratio
        sdf_split_ratio = getattr(args, 'sdf_densify_ratio', 0.1)
        n_sdf_split = int(model.num_int_verts * sdf_split_ratio)

        if n_sdf_split > 0:
            # 计算每个四面体的外接圆半径
            tet_vertices = model.vertices[model.indices]  # (N_tets, 4, 3)
            from utils.topo_utils import calculate_circumcenters_torch
            _, circumradii = calculate_circumcenters_torch(tet_vertices.double())
            circumradii = circumradii.float().squeeze(-1) if circumradii.dim() > 1 else circumradii.float()

            # 获取每个四面体的顶点 SDF
            vert_sdf = model.query_vert_sdf()  # (N_verts,)
            tet_sdf = vert_sdf[model.indices]  # (N_tets, 4)

            # 判断是否跨越表面：四面体中有异号 SDF 值（比 threshold 更稳健）
            tet_sdf_max = tet_sdf.max(dim=1).values  # (N_tets,)
            tet_sdf_min = tet_sdf.min(dim=1).values  # (N_tets,)
            surface_mask = (tet_sdf_max > 0) & (tet_sdf_min < 0)  # 有正有负 = 跨越零等值面

            # 排除已经被误差驱动分裂选中的（避免重复）
            surface_mask = surface_mask & (~error_driven_mask)

            if surface_mask.sum() > 0:
                # 从跨越表面的四面体中，选择外接圆最大的 TopK
                surface_radii = circumradii.clone()
                surface_radii[~surface_mask] = -1  # 非表面四面体设为 -1

                n_sdf_split = min(n_sdf_split, surface_mask.sum().item())
                _, topk_indices = torch.topk(surface_radii, n_sdf_split)
                sdf_driven_mask[topk_indices] = True

                # 计算分裂点 = 四面体重心
                sdf_tet_vertices = tet_vertices[topk_indices]  # (K, 4, 3)
                sdf_split_points = sdf_tet_vertices.mean(dim=1)  # (K, 3) 重心

    # 误差驱动分裂的点位置
    error_split_point, bad = get_approx_ray_intersections(stats.within_var_rays)
    error_split_point = error_split_point[error_driven_mask]
    bad = bad[error_driven_mask]

    # 使用四面体内的随机重心坐标位置作为回退
    error_clone_indices = model.indices[error_driven_mask]
    if error_clone_indices.shape[0] > 0:
        barycentric = torch.rand((error_clone_indices.shape[0], 4, 1), device=device).clip(min=0.1, max=0.9)
        barycentric_weights = barycentric / barycentric.sum(dim=1, keepdim=True)
        random_locations = (model.vertices[error_clone_indices] * barycentric_weights).sum(dim=1)
        error_split_point[bad] = random_locations[bad]

    # 合并分裂点
    if sdf_split_points is not None and sdf_split_points.shape[0] > 0:
        split_point = torch.cat([error_split_point, sdf_split_points.float()], dim=0)
    else:
        split_point = error_split_point

    n_total_split = split_point.shape[0]
    n_sdf_split_actual = sdf_split_points.shape[0] if sdf_split_points is not None else 0

    if n_total_split > 0:
        tet_optim.split(split_point, **args.as_dict())

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"#Grow: {total_mask.sum():4d} #Split: {within_mask.sum():4d} #SDF: {n_sdf_split_actual:4d} "
        f"#Total: {n_total_split:5d} | Total Avg: {total_var.mean():.4f} Within Avg: {within_var.mean():.4f}"
    )
