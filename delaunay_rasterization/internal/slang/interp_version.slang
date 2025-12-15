import intersect;
import utils;
import safe_math;

// [SDF-MODIFICATION] 四面体数据结构，包含顶点SDF和渲染参数
struct SplatTetrahedra_interp : IDifferentiable
{
    uint4 vtet;
    Tetrahedra tet;
    float3 base_color;
    float3 grd;
    float sdf;  // [SDF-MODIFICATION] 四面体中心SDF值 (用于颜色积分)
    float4 vert_sdf;  // [SDF-MODIFICATION] 4个顶点的SDF值 (用于NeuS alpha插值)
    // === [DENSITY-RELATED] 原密度字段 (已注释) ===
    // float density;  // [DENSITY-RELATED] 四面体密度值
    // === [DENSITY-RELATED] 原密度代码结束 ===
};

void print_verts(float4x3 verts) {
    printf("%f, %f, %f, %f\n", verts[0].x, verts[1].x, verts[2].x, verts[3].x);
}
void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}

// [SDF-MODIFICATION] 计算点在四面体内的重心坐标
// 返回 float4(lambda0, lambda1, lambda2, lambda3)，使得 point = sum(lambda_i * v_i)
[Differentiable]
inline float4 compute_barycentric(float3 point, float3 v0, float3 v1, float3 v2, float3 v3) {
    // 使用体积比计算重心坐标
    // lambda_i = Volume(point, v_j, v_k, v_l) / Volume(v0, v1, v2, v3)
    float3 r = point - v3;
    float3 e0 = v0 - v3;
    float3 e1 = v1 - v3;
    float3 e2 = v2 - v3;

    // 计算 3x3 矩阵的行列式 [e0, e1, e2]
    float det = dot(e0, cross(e1, e2));
    float inv_det = safe_div(1.0f, det);

    // 使用克莱姆法则求解
    float lambda0 = dot(r, cross(e1, e2)) * inv_det;
    float lambda1 = dot(e0, cross(r, e2)) * inv_det;
    float lambda2 = dot(e0, cross(e1, r)) * inv_det;
    float lambda3 = 1.0f - lambda0 - lambda1 - lambda2;

    return float4(lambda0, lambda1, lambda2, lambda3);
}

// [SDF-MODIFICATION] 使用重心坐标插值SDF
[Differentiable]
inline float interpolate_sdf(float4 bary, float4 vert_sdf) {
    return bary.x * vert_sdf.x + bary.y * vert_sdf.y + bary.z * vert_sdf.z + bary.w * vert_sdf.w;
}

[Differentiable]
inline float phi(float x) {
    if (abs(x) < 1e-6) {
        return 1.0f - x / 2.0f;
    }
    // return (1.0f - exp(-x)) / x;
    return -(safe_expm1(-x)) / x;
}

[Differentiable]
inline float3 compute_integral(float3 c0, float3 c1, float ddt) {
    float alpha = safe_exp(-ddt);
    
    float phi_val = phi(ddt);
    
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    
    return w0 * c0 + w1 * c1;
}

[Differentiable]
inline CtrlPt integrate_tetra_segment(
    in SplatTetrahedra_interp g,
    in float3 ray_o,
    in float3 ray_d,
    in float enter,
    in float exit,
    in float inv_s,  // [SDF-MODIFICATION] NeuS inv_s 参数 (可微)
)
{
    float dt = max(exit - enter, 0.0f);

    // 计算入口和出口点的世界坐标
    float3 p_enter = enter * ray_d + ray_o;
    float3 p_exit = exit * ray_d + ray_o;

    // 颜色计算 (相对于v0的偏移)
    float3 v_enter = p_enter - g.tet.verts[0];
    float3 v_exit = p_exit - g.tet.verts[0];
    float3 c_enter = dot(g.grd, v_enter) + g.base_color;
    float3 c_exit = dot(g.grd, v_exit) + g.base_color;

    // [SDF-MODIFICATION] NeuS alpha 计算
    // 1. 计算入口和出口点的重心坐标
    float4 bary_enter = compute_barycentric(p_enter, g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);
    float4 bary_exit = compute_barycentric(p_exit, g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    // 2. 插值得到入口和出口点的SDF值
    float sdf_enter = interpolate_sdf(bary_enter, g.vert_sdf);
    float sdf_exit = interpolate_sdf(bary_exit, g.vert_sdf);

    // 3. NeuS CDF: Φ(x) = sigmoid(x * inv_s) = 1 / (1 + exp(-x * inv_s))
    // 使用 safe_exp 避免数值问题
    float cdf_enter = safe_div(1.0f, 1.0f + safe_exp(-sdf_enter * inv_s));
    float cdf_exit = safe_div(1.0f, 1.0f + safe_exp(-sdf_exit * inv_s));

    // 4. NeuS alpha: α = max((Φ(sdf_enter) - Φ(sdf_exit)) / Φ(sdf_enter), 0)
    // 注意：当从外部（sdf>0）进入内部（sdf<0）时，cdf_enter > cdf_exit，alpha > 0
    float alpha = max(safe_div(cdf_enter - cdf_exit, cdf_enter + 1e-10f), 0.0f);
    alpha = safe_clip(alpha, 0.0f, 1.0f);

    // === [DENSITY-RELATED] 原密度公式 (已注释) ===
    // float alpha = max(-safe_expm1(-g.sdf * dt), 0.f);
    // === [DENSITY-RELATED] 原密度代码结束 ===

    // [SDF-MODIFICATION] 颜色积分仍使用四面体中心SDF (g.sdf) 作为密度近似
    // 这是为了保持颜色渐变的平滑性
    float effective_density = max(g.sdf, 0.0f);  // 使用正SDF作为密度
    float3 final_color = compute_integral(c_exit, c_enter, dt * effective_density);

    // 将 alpha 转换为 -log(1-alpha) 以便累积 (与原代码兼容)
    float log_transmittance = -safe_log(max(1.0f - alpha, 1e-10f));

    float4 rgba = {final_color.x * alpha,
                    final_color.y * alpha,
                    final_color.z * alpha,
                    log_transmittance};  // [SDF-MODIFICATION] 使用 log transmittance
    return { rgba, {enter, exit}, effective_density };  // [SDF-MODIFICATION] 返回 effective_density 用于 distortion
}

[Differentiable]
inline bool evaluate_tetra_interp(
    in SplatTetrahedra_interp g,
    in float3 ray_o,
    in float3 ray_d,
    no_diff in float t_min,
    in float inv_s,  // [SDF-MODIFICATION] NeuS inv_s 参数 (可微)
    out CtrlPt ctrl_pt)
{
    float2 dist;

    float4x3 verts = float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    float3 enter_norm, exit_norm;
    bool hit = ray_tetrahedron_intersect_fused(
    // bool hit = ray_tetrahedron_intersect_plucker(
        ray_o, ray_d, verts, dist, enter_norm, exit_norm);

    float enter = dist.x;
    float exit = max(dist.y, enter);

    if (!hit) {
        ctrl_pt = { float4(0.f, 0.f, 0.f, 0.f), float2(0.f, 0.f), 0.f };  // [SDF-MODIFICATION]
        return false;
    } else {
        ctrl_pt = integrate_tetra_segment(g, ray_o, ray_d, enter, exit, inv_s);
        return true;
    }
}

// [SDF-MODIFICATION] 加载四面体数据（可微分版本）
[Differentiable]
inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView tet_sdf,
    DiffTensorView vert_sdf)  // [SDF-MODIFICATION] 全局顶点SDF数组
{
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    // [SDF-MODIFICATION] 从 tet_sdf 张量加载四面体中心SDF值 (索引0)
    float sdf = float(tet_sdf[g_idx, 0]);  // [SDF-MODIFICATION] SDF可以是负数，不需要max
    float3 base_color = {
        float(tet_sdf[g_idx, 1]),
        float(tet_sdf[g_idx, 2]),
        float(tet_sdf[g_idx, 3]),
    };
    float3 grd = {
        float(tet_sdf[g_idx, 4]),
        float(tet_sdf[g_idx, 5]),
        float(tet_sdf[g_idx, 6]),
    };

    // [SDF-MODIFICATION] 从全局顶点SDF数组加载4个顶点的SDF值
    float4 vert_sdf_values = {
        float(vert_sdf[virtual_tet.x, 0]),
        float(vert_sdf[virtual_tet.y, 0]),
        float(vert_sdf[virtual_tet.z, 0]),
        float(vert_sdf[virtual_tet.w, 0]),
    };

    SplatTetrahedra_interp result = {
        virtual_tet, tet, base_color, grd, sdf, vert_sdf_values  // [SDF-MODIFICATION] 包含顶点SDF
    };
    return result;
}

// [SDF-MODIFICATION] 加载四面体数据（非可微分版本）
inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  tet_sdf,
    TensorView<float>  vert_sdf)  // [SDF-MODIFICATION] 全局顶点SDF数组
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    // [SDF-MODIFICATION] 从 tet_sdf 张量加载四面体中心SDF值 (索引0)
    float sdf = float(tet_sdf[g_idx, 0]);  // [SDF-MODIFICATION] SDF可以是负数
    SplatTetrahedra_interp result;
    result.vtet = virtual_tet;
    result.tet       = tet;
    result.sdf       = sdf;  // [SDF-MODIFICATION] 设置sdf字段
    result.base_color = {
        float(tet_sdf[g_idx, 1]),
        float(tet_sdf[g_idx, 2]),
        float(tet_sdf[g_idx, 3]),
    };
    result.grd = {
        float(tet_sdf[g_idx, 4]),
        float(tet_sdf[g_idx, 5]),
        float(tet_sdf[g_idx, 6]),
    };
    // [SDF-MODIFICATION] 从全局顶点SDF数组加载4个顶点的SDF值
    result.vert_sdf = {
        float(vert_sdf[virtual_tet.x, 0]),
        float(vert_sdf[virtual_tet.y, 0]),
        float(vert_sdf[virtual_tet.z, 0]),
        float(vert_sdf[virtual_tet.w, 0]),
    };
    return result;
}
