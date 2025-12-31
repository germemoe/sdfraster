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

// [NORMAL-OUTPUT] 计算四面体内 SDF 线性场的梯度（法向量）
// 由于四面体内 SDF 是通过重心坐标线性插值的，梯度为常值
// ∇sdf = s0 * ∇λ0 + s1 * ∇λ1 + s2 * ∇λ2 + s3 * ∇λ3
// 其中 ∇λi 是重心坐标的梯度，可从四面体几何计算
[Differentiable]
inline float3 compute_tet_sdf_gradient(float3 v0, float3 v1, float3 v2, float3 v3, float4 vert_sdf) {
    // 计算四面体边向量
    float3 e0 = v0 - v3;
    float3 e1 = v1 - v3;
    float3 e2 = v2 - v3;

    // 计算四面体体积的 6 倍（带符号）
    float det = dot(e0, cross(e1, e2));
    float inv_det = safe_div(1.0f, det);

    // 重心坐标的梯度
    // ∇λ0 = (v1-v3) × (v2-v3) / (6V) = cross(e1, e2) / det
    // ∇λ1 = (v2-v3) × (v0-v3) / (6V) = cross(e2, e0) / det
    // ∇λ2 = (v0-v3) × (v1-v3) / (6V) = cross(e0, e1) / det
    // ∇λ3 = -∇λ0 - ∇λ1 - ∇λ2
    float3 grad_lambda0 = cross(e1, e2) * inv_det;
    float3 grad_lambda1 = cross(e2, e0) * inv_det;
    float3 grad_lambda2 = cross(e0, e1) * inv_det;
    float3 grad_lambda3 = -grad_lambda0 - grad_lambda1 - grad_lambda2;

    // SDF 梯度 = Σ sdf_i * ∇λi
    float3 sdf_gradient = vert_sdf.x * grad_lambda0
                        + vert_sdf.y * grad_lambda1
                        + vert_sdf.z * grad_lambda2
                        + vert_sdf.w * grad_lambda3;

    return sdf_gradient;
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

// [SDF-ALPHA-MODE] 开关：true = 使用中点+梯度计算出入点SDF（NeuS风格），false = 直接插值出入点SDF
static const bool USE_GRADIENT_SDF = false;

[Differentiable]
inline CtrlPt integrate_tetra_segment(
    in SplatTetrahedra_interp g,
    in float3 ray_o,
    in float3 ray_d,
    in float enter,
    in float exit,
    in float inv_s,
)
{
    float dt = max(exit - enter, 0.0f);

    float3 p_enter = enter * ray_d + ray_o;
    float3 p_exit  = exit  * ray_d + ray_o;

    // 颜色仍然线性（你的 base_color + grd）
    float3 v_enter = p_enter - g.tet.verts[0];
    float3 v_exit  = p_exit  - g.tet.verts[0];
    float3 c_enter = dot(g.grd, v_enter) + g.base_color;  // c0
    float3 c_exit  = dot(g.grd, v_exit)  + g.base_color;  // c1

    // [SDF-ALPHA-MODE] 计算四面体内SDF梯度（常值），后面复用于normal计算
    float3 sdf_grad = compute_tet_sdf_gradient(
        g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3],
        g.vert_sdf);

    float sdf_enter, sdf_exit;

    if (USE_GRADIENT_SDF) {
        // [SDF-ALPHA-MODE] 新方式：使用中点SDF + 梯度计算出入点SDF
        // 参考 NeuS renderer.py 第237-247行
        float3 p_mid = (p_enter + p_exit) * 0.5f;

        // 计算中点的重心坐标和SDF
        float4 bary_mid = compute_barycentric(p_mid, g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);
        float sdf_mid = interpolate_sdf(bary_mid, g.vert_sdf);

        // 沿射线方向的SDF变化量（无需归一化）
        // 数学上: dot(∇sdf, ray_d/|ray_d|) * (dt * |ray_d|) = dot(∇sdf, ray_d) * dt
        // 归一化和距离缩放相互抵消，可直接计算
        float sdf_delta = dot(sdf_grad, ray_d) * dt * 0.5f;

        // 根据中点SDF和变化量估算出入点SDF
        sdf_enter = sdf_mid - sdf_delta;
        sdf_exit  = sdf_mid + sdf_delta;
    } else {
        // 原方式：直接插值出入点的SDF
        float4 bary_enter = compute_barycentric(p_enter, g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);
        float4 bary_exit  = compute_barycentric(p_exit , g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

        sdf_enter = interpolate_sdf(bary_enter, g.vert_sdf);
        sdf_exit  = interpolate_sdf(bary_exit , g.vert_sdf);
    }

    // Φ = sigmoid(sdf * inv_s)
    float u0 = sdf_enter * inv_s;
    float u1 = sdf_exit  * inv_s;
    float Phi0 = safe_sigmoid(u0);   // cdf_enter
    float Phi1 = safe_sigmoid(u1);   // cdf_exit

    // alpha = 1 - Phi1/Phi0  (与 (Phi0-Phi1)/Phi0 等价)
    float denom0 = max(Phi0, 1e-10f);
    float alpha = 1.0f - safe_div(Phi1, denom0);
    alpha = safe_clip(alpha, 0.0f, 1.0f);

    // 计算 J = ∫_0^1 Phi(tau) d tau = (softplus(u1)-softplus(u0)) / (u1-u0)
    float du = u1 - u0;
    float J;
    if (abs(du) > 1e-6f) {
        float sp0 = safe_softplus(u0);
        float sp1 = safe_softplus(u1);
        J = safe_div(sp1 - sp0, du);
    } else {
        // du -> 0: J -> Phi0
        J = Phi0;
    }

    // c_eff = (c0*Phi0 - c1*Phi1 + (c1-c0)*J) / (Phi0 - Phi1)
    float denomPhi = Phi0 - Phi1;
    float3 c_eff;
    if (abs(denomPhi) > 1e-10f) {
        float3 dc = c_exit - c_enter;
        float3 numer = c_enter * Phi0 - c_exit * Phi1 + dc * J;
        c_eff = numer * safe_div(1.0f, denomPhi);
    } else {
        // Phi0≈Phi1: 权重几乎为0，段内颜色取中点即可
        c_eff = 0.5f * (c_enter + c_exit);
    }

    // 段 RGB 贡献（外层再乘当前 T）
    float3 seg_rgb = c_eff * alpha;

    // 光学厚度 tau = -log(1-alpha)（用于外层更新 T，也兼容你原来的 rgba.w 用法）
    float tau = -safe_log(max(1.0f - alpha, 1e-10f));

    // 如果你需要一个"等效密度"给 distortion：rho_hat = tau / dt
    float rho_hat = (dt > 1e-8f) ? (tau / dt) : 0.0f;

    // [NORMAL-OUTPUT] 复用上面已计算的 sdf_grad 作为法向量
    float grad_len = length(sdf_grad);
    float3 normal = (grad_len > 1e-8f)
        ? sdf_grad / grad_len
        : float3(0.0f, 0.0f, 1.0f);

    float4 rgba = { seg_rgb.x, seg_rgb.y, seg_rgb.z, tau };
    return { rgba, {enter, exit}, rho_hat, normal };
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
        ctrl_pt = { float4(0.f, 0.f, 0.f, 0.f), float2(0.f, 0.f), 0.f, float3(0.f, 0.f, 0.f) };  // [NORMAL-OUTPUT] 添加 normal 零值
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
