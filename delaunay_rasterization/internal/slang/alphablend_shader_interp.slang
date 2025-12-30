// Copyright 2025 Alexander Mai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import utils;
import camera;
import safe_math;
import interp_version;

static const uint TILE_HEIGHT = PYTHON_TILE_HEIGHT;
static const uint TILE_WIDTH = PYTHON_TILE_WIDTH;

groupshared SplatTetrahedra_interp collected_splats[TILE_HEIGHT * TILE_WIDTH];
groupshared uint32_t collected_idx[TILE_HEIGHT * TILE_WIDTH];

struct DistortionState5 : IDifferentiable {
    float x;  // old_state.x
    float y;  // old_state.y
    float z;  // old_state.z
    float w;  // old_state.w
    float v;  // new 5th term, e.g., for w_i^2 * interval / 3, or anything else
};

[Differentiable]
float tukey_power_ladder(float x, float p) {
  // Compute sign(x) * |p - 1|/p * ((|x|/|p-1| + 1)^p - 1)
  if (p == 1) {
    return x;
  }
  let xp = abs(x);
  let xs = xp / max(TINY_VAL, abs(p - 1));
  let y = sign(x) * abs(p - 1) / p * (pow(xs + 1, p) - 1);
  return y;
}

[Differentiable]
float safe_exp4alpha(float x) {
    return safe_clip(safe_exp(x), 0, 1);
}

// [SDF-MODIFICATION] 辅助函数：从 DiffTensorView 读取 inv_s
// 使用单独的函数使得 bwd_diff 可以自动处理梯度累积
[Differentiable]
float read_inv_s(DiffTensorView inv_s_tensor) {
    return inv_s_tensor[uint(0)];
}

// [Differentiable]
// float moment_0_int(float a, float b, float s1, float T) {
//     return T * safe_div((-safe_expm1(s1*(a - b))),s1);
// }
// [Differentiable]
// float moment_1_int(float a, float b, float s1, float T) {
//     float s12 = s1*s1;
//     if (s12 < 1e-6) {
//         return T * 0.5 * max(b * b - a * a, 0);
//     } else {
//         // return (a*s1 - (b*s1 + 1)*safe_exp4alpha(s1*(a - b)) + 1) / (s1*s1);
//         return T * max(
//             - safe_div(safe_expm1(s1*(a - b)), s12)
//             + safe_div(a - b*safe_exp(s1*(a - b)), s1), 0);
//     }
// }
// [Differentiable]
// float self_dist(float a, float b, float s, float T) {
//     float dt = (a - b);
//     float ds = s * dt;
//     float s3 = s*s*s;
//     if (abs(ds) < 1e-4 || s3 < 1e-5) {
//         return T * T * max(-1 / 3 * pow(dt, 3), 0);
//     } else {
//         return T * T * max(
//             safe_div(
//                 ( 2 * s * (a - b) * exp(ds) - safe_expm1(2 * ds)),
//                 s3),
//             0);
//     }
// }

[Differentiable]
float moment_0_int(float a, float b, float s1, float T) {
    return s1;
}
[Differentiable]
float moment_1_int(float a, float b, float s1, float T) {
    return s1 * (b+a)/2;
}
[Differentiable]
float self_dist(float a, float b, float s, float T) {
    float dt = max(b - a, 0);
    return s*s*dt/3;
}

[Differentiable]
DistortionState5 update_distortion_state(
    DistortionState5 old_state,
    float T,
    float density,
    float2 dist)
{
    // no_diff float start = tukey_power_ladder(500*dist.x, -0.1);
    // no_diff float end = tukey_power_ladder(500*dist.y, -0.1);
    // float start = (2*dist.x) / (2+dist.x);
    // float end = (2*dist.y) / (2+dist.y);
    float start = dist.x;
    float end = dist.y;
    float scaling = 1;
    float d = density;

    float m0 = moment_0_int(start, end, d, T); // just weight
    float m1 = moment_1_int(start, end, d, T);

    DistortionState5 new_state;
    new_state.x = old_state.x + 2.0f * m1 * old_state.z;
    new_state.y = old_state.y + 2.0f * m0 * old_state.w;
    new_state.z = old_state.z + m0;
    new_state.w = old_state.w + m1;
    // new_state.v = old_state.v + T * T * self_dist(start, end, d);
    new_state.v = old_state.v + self_dist(start, end, d, T);
    // printf("%f - %f, d: %f, v: %f T: %f, m0: %f, m1: %f\n", start, end, d, new_state.v, T, m0, m1);

    return new_state;
}

DistortionState5 undo_distortion_state(
    DistortionState5 new_state,
    float T,
    float density,
    float2 dist)
{
    float start = dist.x;
    float end = dist.y;
    float scaling = 1;
    float d = density;

    float m0 = moment_0_int(start, end, d, T);
    float m1 = moment_1_int(start, end, d, T);

    DistortionState5 old_state;
    old_state.z = new_state.z - m0;
    old_state.w = new_state.w - m1;
    old_state.x = new_state.x - 2.0f * m1 * old_state.z;
    old_state.y = new_state.y - 2.0f * m0 * old_state.w;
    old_state.v = new_state.v - self_dist(start, end, d, T);
    return old_state;
}

#define LOG_EPS 1e-6f

[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = safe_exp(pixel_state_t_nm1.a);
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * transmittance_t_nm1;
    float log_transmittance_t_n = pixel_state_t_nm1.a - gauss_rgba_t_n.a;
    
    return float4(color_t_n, log_transmittance_t_n);
}

float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float log_transmittance_t_nm1 = pixel_state_t_n.a + gauss_rgba_t_n.a;
    float transmittance_t_nm1 = safe_exp(log_transmittance_t_nm1);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, log_transmittance_t_nm1);
}

struct AlphaOut: IDifferentiable {
    float4 rgba;
    DistortionState5 distortion_state;
    float3 normal_state;  // [NORMAL-OUTPUT] 累积的 normal（alpha blending）
    float depth_state;    // [DEPTH-OUTPUT] 累积的期望深度
};

// [NORMAL-OUTPUT] Normal blending: accumulated_normal += normal * alpha * T
[Differentiable]
float3 update_normal_state(float3 prev_normal, float3 ctrl_normal, float4 ctrl_rgba, float T) {
    // ctrl_rgba.rgb 已经是 color * alpha，所以这里直接用 alpha
    // alpha = 1 - exp(-tau)，tau 存在 ctrl_rgba.a
    float alpha = 1.0f - safe_exp(-ctrl_rgba.a);
    return prev_normal + ctrl_normal * alpha * T;
}

float3 undo_normal_state(float3 curr_normal, float3 ctrl_normal, float4 ctrl_rgba, float T) {
    float alpha = 1.0f - safe_exp(-ctrl_rgba.a);
    return curr_normal - ctrl_normal * alpha * T;
}

// [DEPTH-OUTPUT] Expected depth: depth += T * alpha * t_mid
// [T-TO-Z-SVRASTER] 在 t 空间累加，最后写入时才乘 ray.d.z（仿 svraster）
[Differentiable]
float update_depth_state(float prev_depth, float4 ctrl_rgba, float2 ctrl_dist, float T) {
    float alpha = 1.0f - safe_exp(-ctrl_rgba.a);
    float t_mid = (ctrl_dist.x + ctrl_dist.y) * 0.5f;
    return prev_depth + T * alpha * t_mid;  // 累加 t 空间深度
}

float undo_depth_state(float curr_depth, float4 ctrl_rgba, float2 ctrl_dist, float T) {
    float alpha = 1.0f - safe_exp(-ctrl_rgba.a);
    float t_mid = (ctrl_dist.x + ctrl_dist.y) * 0.5f;
    return curr_depth - T * alpha * t_mid;  // 回退 t 空间深度
}

[Differentiable]
AlphaOut update_alpha_out(AlphaOut prev, CtrlPt ctrl) {
    float T = safe_exp(prev.rgba.a);
    return {
        update_pixel_state(prev.rgba, ctrl.rgba),
        update_distortion_state(prev.distortion_state, T, ctrl.density, ctrl.dist),
        update_normal_state(prev.normal_state, ctrl.normal, ctrl.rgba, T),  // [NORMAL-OUTPUT]
        update_depth_state(prev.depth_state, ctrl.rgba, ctrl.dist, T)  // [DEPTH-OUTPUT] [T-TO-Z-SVRASTER]
    };
}

AlphaOut undo_alpha_out(AlphaOut curr, CtrlPt ctrl) {
    float4 pix_state = undo_pixel_state(curr.rgba, ctrl.rgba);
    float T = safe_exp(pix_state.a);
    return {
        pix_state,
        undo_distortion_state(curr.distortion_state, T, ctrl.density, ctrl.dist),
        undo_normal_state(curr.normal_state, ctrl.normal, ctrl.rgba, T),  // [NORMAL-OUTPUT]
        undo_depth_state(curr.depth_state, ctrl.rgba, ctrl.dist, T)  // [DEPTH-OUTPUT] [T-TO-Z-SVRASTER]
    };
}

[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
AlphaOut alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   TensorView<int32_t> indices,
                   DiffTensorView vertices,
                   DiffTensorView tet_density,
                   DiffTensorView vert_sdf,  // [SDF-MODIFICATION] 全局顶点SDF数组
                   DiffTensorView inv_s_tensor,  // [SDF-MODIFICATION] NeuS inv_s 参数 (可微tensor)
                   DiffTensorView final_pixel_state,
                   DiffTensorView final_distortion_state,
                   DiffTensorView final_normal_state,  // [NORMAL-OUTPUT] 最终 normal 状态
                   DiffTensorView final_depth_state,   // [DEPTH-OUTPUT] 最终深度状态
                   TensorView<float> depth_med_img,    // [DEPTH-MED] 中位深度输出（不可微）
                   TensorView<int32_t> n_contributors,
                   TensorView<bool> tet_alive,
                   no_diff Camera cam,
                   no_diff float ray_jitter_x,
                   no_diff float ray_jitter_y,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end)
{
    float2 center_pix_coord = {pix_coord.x+ray_jitter_x, pix_coord.y+ray_jitter_y};
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 0.f);
    DistortionState5 curr_distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    float3 curr_normal_state = float3(0.f, 0.f, 0.f);  // [NORMAL-OUTPUT] 初始化 normal 状态
    float curr_depth_state = 0.f;  // [DEPTH-OUTPUT] 初始化深度状态
    AlphaOut ao = {
        curr_pixel_state, curr_distortion_state, curr_normal_state, curr_depth_state
    };
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    //bool print = thread_rank == 0;
    bool print = false;

    Ray ray = no_diff get_ray(cam, center_pix_coord);  // no_diff: 此调用不需要梯度
    int exit_face;
    // if (print) {
    //     printf("ray_o: %f, %f, %f, ray_d: %f, %f, %f\n", ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z);
    // }

    // [DEPTH-MED] 中位深度追踪变量（不可微）
    float depth_med = 0.f;
    float T_med = 1.f;
    bool found_med = false;

    int32_t local_n_contrib = 0;
    // int32_t low = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            // [MEMORY-GUARD] 边界检查，防止越界访问
            if (coll_id >= indices.size(0)) {
                coll_id = 0;
            }
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend_interp(coll_id, vertices, indices, tet_density, vert_sdf);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra_interp g = collected_splats[j];
                CtrlPt tetra_ctrl;
                // [SDF-MODIFICATION] 使用 read_inv_s 读取 inv_s
                float inv_s = read_inv_s(inv_s_tensor);
                if (evaluate_tetra_interp(g, ray.o, ray.d, cam.min_t, inv_s, tetra_ctrl) && tetra_ctrl.rgba.a > 0) {
                    AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl);  // [T-TO-Z-SVRASTER] 去掉 ray.d.z

                    // [DEPTH-MED] 追踪中位深度：当 T 从 >0.5 变为 <=0.5 时记录
                    if (!found_med) {
                        float alpha_seg = 1.0f - exp(-tetra_ctrl.rgba.a);
                        float new_T_med = T_med * (1.0f - alpha_seg);
                        if (T_med > 0.5f && new_T_med <= 0.5f) {
                            float t_mid = (tetra_ctrl.dist.x + tetra_ctrl.dist.y) * 0.5f;
                            depth_med = t_mid;  // [T-TO-Z-SVRASTER] 保持 t 空间，最后才转换
                            found_med = true;
                        }
                        T_med = new_T_med;
                    }

                    ao = new_ao;
                    // if (new_ao.rgba.a < 1./255.f) {
                    if (new_ao.rgba.a < -5.5) {
                        thread_active = false;
                        break;
                    }
                }
            }
        }
        splats_left_to_process -= block_size;
    }

    if (is_inside) {
        n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = local_n_contrib;
        // [DEPTH-MED] [T-TO-Z-SVRASTER] 写入中位深度时才除 ||d||，仿 svraster
        float rd_norm_inv = 1.0f / length(ray.d);
        depth_med_img[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))] = depth_med * rd_norm_inv;
    }

    return ao;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     TensorView<int32_t> indices,
                     DiffTensorView vertices,
                     DiffTensorView tet_density,
                     DiffTensorView vert_sdf,  // [SDF-MODIFICATION] 全局顶点SDF数组
                     DiffTensorView inv_s_tensor,  // [SDF-MODIFICATION] NeuS inv_s 参数 (可微tensor)
                     DiffTensorView final_pixel_state,
                     DiffTensorView final_distortion_state,
                     DiffTensorView final_normal_state,  // [NORMAL-OUTPUT] 最终 normal 状态
                     DiffTensorView final_depth_state,   // [DEPTH-OUTPUT] 最终深度状态
                     TensorView<float> depth_med_img,    // [DEPTH-MED] 中位深度（不可微，反向不处理）
                     TensorView<int32_t> n_contributors,
                     TensorView<bool> tet_alive,
                     no_diff Camera cam,
                     no_diff float ray_jitter_x,
                     no_diff float ray_jitter_y,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     AlphaOut d_ao)
{
    float2 center_pix_coord = {pix_coord.x+ray_jitter_x, pix_coord.y+ray_jitter_y};
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    uint32_t block_size = cam.tile_height * cam.tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);

    int splats_left_to_process = tile_idx_end - tile_idx_start;
    int64_t current_splat_offset = tile_idx_end - tile_idx_start;

    float4 current_pixel_state;
    DistortionState5 curr_distortion_state;
    float3 curr_normal_state;  // [NORMAL-OUTPUT]
    float curr_depth_state;    // [DEPTH-OUTPUT]
    int32_t n_contrib_fwd;
    if (is_inside) {
        curr_distortion_state = {final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)],
                                 final_distortion_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4)]};
        current_pixel_state = float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        // [NORMAL-OUTPUT] 从 final_normal_state 读取 normal 状态
        curr_normal_state = float3(final_normal_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                   final_normal_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                   final_normal_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)]);
        // [DEPTH-OUTPUT] 从 final_depth_state 读取深度状态
        // [DEPTH-FIX] 前向 splat_tiled 存储时乘了 rd_norm_inv（转为 z-space）
        // 这里需要乘回 ray.d 长度，恢复为 t-space 以正确回退状态
        curr_depth_state = final_depth_state[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))];
        n_contrib_fwd = n_contributors[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }

    Ray ray = get_ray(cam, center_pix_coord);  // 普通函数，不需要 no_diff
    int exit_face;

    // [DEPTH-FIX] 将 z-space 深度转回 t-space
    // 前向 splat_tiled 存储时乘了 rd_norm_inv，这里需要乘回 ray.d 长度
    if (is_inside) {
        curr_depth_state = curr_depth_state * length(ray.d);
    }
    AlphaOut ao = {current_pixel_state, curr_distortion_state, curr_normal_state, curr_depth_state};

    DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);

    // [SDF-MODIFICATION] 从 tensor 读取 inv_s
    float inv_s = read_inv_s(inv_s_tensor);

    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            // [MEMORY-GUARD] 边界检查，防止越界访问
            if (coll_id >= indices.size(0)) {
                coll_id = 0;
            }
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend_interp(coll_id, vertices, indices, tet_density, vert_sdf);
        }
        AllMemoryBarrierWithGroupSync();
        if (is_inside) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                current_splat_offset--;
                if (current_splat_offset >= n_contrib_fwd || current_splat_offset < 0)
                    continue;
                uint32_t g_idx = collected_idx[j];
                SplatTetrahedra_interp g = collected_splats[j];

                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray.o, ray.d, cam.min_t, inv_s, tetra_ctrl) && tetra_ctrl.rgba.a > 0) {
                    ao = undo_alpha_out(ao, tetra_ctrl);  // [T-TO-Z-SVRASTER] 去掉 ray.d.z

                    // Back-prop automatically through blending and gaussian evaluation.
                    DifferentialPair<SplatTetrahedra_interp> dp_g = diffPair(g);
                    DifferentialPair<float3> dp_ray_o = diffPair(ray.o);
                    DifferentialPair<float3> dp_ray_d = diffPair(ray.d);
                    DifferentialPair<CtrlPt> dp_tetra_ctrl = diffPair(tetra_ctrl);
                    DifferentialPair<AlphaOut> dp_ao = diffPair(ao);
                    // [SDF-MODIFICATION] 创建 inv_s 的 DifferentialPair
                    DifferentialPair<float> dp_inv_s = diffPair(inv_s, 0.0f);

                    bwd_diff(update_alpha_out)(dp_ao, dp_tetra_ctrl, d_ao);  // [T-TO-Z-SVRASTER] 去掉 ray.d.z
                    d_ao = dp_ao.getDifferential();
                    float midpt = tetra_ctrl.dist.x;
                    // [NORMAL-OUTPUT] 添加 normal 梯度回传
                    CtrlPt.Differential d_tetra_scale = {
                        dp_tetra_ctrl.d.rgba,
                        dp_tetra_ctrl.d.dist,
                        dp_tetra_ctrl.d.density,
                        dp_tetra_ctrl.d.normal,
                    };
                    // [SDF-MODIFICATION] 传递 inv_s DifferentialPair 到 bwd_diff
                    bwd_diff(evaluate_tetra_interp)(dp_g, dp_ray_o, dp_ray_d, cam.min_t, dp_inv_s, d_tetra_scale);
                    // [SDF-MODIFICATION] 使用 bwd_diff(read_inv_s) 累积 inv_s 梯度到 tensor
                    bwd_diff(read_inv_s)(inv_s_tensor, dp_inv_s.getDifferential());
                    // [SDF-MODIFICATION] 传递 vert_sdf 到 load_tet_alphablend_interp
                    bwd_diff(load_tet_alphablend_interp)(g_idx, vertices, indices, tet_density, vert_sdf, dp_g.d);

                }
            }
        }
        splats_left_to_process -= block_size;
    }
    // [SDF-MODIFICATION] inv_s 梯度通过 bwd_diff(read_inv_s) 累积到 tensor
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 TensorView<int32_t> indices,
                 DiffTensorView vertices,
                 DiffTensorView tet_density,
                 DiffTensorView vert_sdf,  // [SDF-MODIFICATION] 全局顶点SDF数组
                 DiffTensorView inv_s_tensor,  // [SDF-MODIFICATION] NeuS inv_s 参数 (可微tensor)
                 DiffTensorView output_img,
                 DiffTensorView distortion_img,
                 DiffTensorView normal_img,  // [NORMAL-OUTPUT] normal 输出
                 DiffTensorView depth_img,   // [DEPTH-OUTPUT] 深度输出
                 TensorView<float> depth_med_img,  // [DEPTH-MED] 中位深度输出（不可微）
                 TensorView<int32_t> n_contributors,
                 TensorView<bool> tet_alive,
                 TensorView<float> ray_jitter,
                 TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;
    Camera cam = no_diff load_tensor_camera(tcam);

    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float rjx = 0;
    float rjy = 0;
    if (is_inside) {
        rjx = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(0))];
        rjy = no_diff ray_jitter[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), uint32_t(1))];
    }

    AlphaOut pixel_states = alpha_blend(
        sorted_gauss_idx,
        indices,
        vertices,
        tet_density,
        vert_sdf,          // [SDF-MODIFICATION] 全局顶点SDF数组
        inv_s_tensor,      // [SDF-MODIFICATION] NeuS inv_s 参数 (可微tensor)
        output_img,
        distortion_img,
        normal_img,        // [NORMAL-OUTPUT]
        depth_img,         // [DEPTH-OUTPUT]
        depth_med_img,     // [DEPTH-MED] 中位深度
        n_contributors,
        tet_alive,
        cam,
        rjx,
        rjy,
        pix_coord,
        tile_idx_start,
        tile_idx_end);

    // [T-TO-Z-SVRASTER] 获取 ray 用于深度 t→z 转换
    float2 center_pix_coord = {pix_coord.x+rjx, pix_coord.y+rjy};
    Ray ray = no_diff get_ray(cam, center_pix_coord);  // no_diff: 此调用不需要梯度
    float rd_norm_inv = 1.0f / length(ray.d);  // 仿 svraster: z = t / ||d||

    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.rgba.r);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.rgba.g);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.rgba.b);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_states.rgba.a);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.distortion_state.x);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.distortion_state.y);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.distortion_state.z);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_states.distortion_state.w);
      distortion_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 4), pixel_states.distortion_state.v);
      // [NORMAL-OUTPUT] 写入 normal 输出
      normal_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_states.normal_state.x);
      normal_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_states.normal_state.y);
      normal_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_states.normal_state.z);
      // [DEPTH-OUTPUT] [T-TO-Z-SVRASTER] 写入深度时才除 ||d||，仿 svraster
      depth_img.storeOnce(uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x)), pixel_states.depth_state * rd_norm_inv);
    }
}

void kern_calc_tet_err(TensorView<int32_t> sorted_gauss_idx,
                  TensorView<int32_t> indices,
                  TensorView<float> vertices,
                  TensorView<float> tet_density,
                  TensorView<float> vert_sdf,  // [SDF-MODIFICATION] 全局顶点SDF数组
                  float inv_s,                 // [SDF-MODIFICATION] NeuS inv_s 参数
                  TensorView<float> final_pixel_state,
                  TensorView<float> tet_err,
                  TensorView<int32_t> tet_count,
                  TensorView<int32_t> n_contributors,
                  Camera cam,
                  uint32_t2 pix_coord,
                  uint32_t tile_idx_start,
                  uint32_t tile_idx_end,
                  float pixel_err,
                  float ssim_err,
                  float3 gt_pixel)
{
    float2 center_pix_coord = {pix_coord.x+0.5, pix_coord.y+0.5};
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    DistortionState5 curr_distortion_state = {0.f, 0.f, 0.f, 0.f, 0.f};
    float3 curr_normal_state = float3(0.f, 0.f, 0.f);  // [NORMAL-OUTPUT]
    float curr_depth_state = 0.f;  // [DEPTH-OUTPUT]
    AlphaOut ao = {
        curr_pixel_state, curr_distortion_state, curr_normal_state, curr_depth_state
    };
    uint32_t block_size = cam.tile_height * cam.tile_width;
    bool is_inside = (pix_coord.x < cam.W && pix_coord.y < cam.H);
    bool thread_active = is_inside;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start) + block_size - 1) / block_size;
    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;

    Ray ray = get_ray(cam, center_pix_coord);  // 普通函数，不需要 no_diff
    int exit_face;

    float max_split_alpha = 0.f;
    float max_grow_alpha = 0.f;

    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int splat_pointer_offset = i * block_size + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            // [MEMORY-GUARD] 边界检查，防止越界访问
            if (coll_id >= indices.size(0)) {
                coll_id = 0;
            }
            collected_idx[thread_rank] = coll_id;
            collected_splats[thread_rank] = load_tet_alphablend_interp(coll_id, vertices, indices, tet_density, vert_sdf);
        }
        AllMemoryBarrierWithGroupSync();
        if (thread_active) {
            for (int j = 0; j < min(block_size, splats_left_to_process); j++)
            {
                local_n_contrib++;
                SplatTetrahedra_interp g = collected_splats[j];
                uint32_t g_idx = collected_idx[j];
                CtrlPt tetra_ctrl;
                if (evaluate_tetra_interp(g, ray.o, ray.d, cam.min_t, inv_s, tetra_ctrl) && tetra_ctrl.rgba.a > 0) {
                    AlphaOut new_ao = update_alpha_out(ao, tetra_ctrl);  // [T-TO-Z-SVRASTER] 去掉 ray.d.z

                    float trans = exp(ao.rgba.a);
                    float T = trans * tetra_ctrl.rgba.a;
                    // float T = safe_exp(new_ao.rgba.a);
                    // float3 diff = new_ao.rgba.rgb - gt_pixel;
                    // float mismatch = (diff.x + diff.y + diff.z)/3;
                    float mismatch = pixel_err;

                    float old_val;
                    int32_t old_ival;

                    tet_err.InterlockedAdd(uint2(g_idx, 0), T, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 1), T * mismatch, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 2), T * mismatch * mismatch, old_val);
                    // tet_err.InterlockedAdd(uint2(g_idx, 3), 1, old_val);
                    // tet_err.InterlockedAdd(uint2(g_idx, 4), T * abs(pixel_err), old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 5), T * ssim_err, old_val);

                    float w = T;
                    float3 p1 = tetra_ctrl.dist.x * ray.d + ray.o;
                    float3 p2 = tetra_ctrl.dist.y * ray.d + ray.o;
                    tet_err.InterlockedAdd(uint2(g_idx, 6), w * p1.x, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 7), w * p1.y, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 8), w * p1.z, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 9), w * p2.x, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 10), w * p2.y, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 11), w * p2.z, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 12), w, old_val);

                    float3 centroid = 0.5*(tetra_ctrl.dist.x+tetra_ctrl.dist.y) * ray.d + ray.o;
                    tet_err.InterlockedAdd(uint2(g_idx, 13), w * centroid.x, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 14), w * centroid.y, old_val);
                    tet_err.InterlockedAdd(uint2(g_idx, 15), w * centroid.z, old_val);

                    tet_count.InterlockedAdd(uint2(g_idx, 0), 1, old_ival);
                    int T_int = int(65535.0 * T);
                    tet_count.InterlockedMax(uint2(g_idx, 1), T_int, old_ival);

                    ao = new_ao;
                    if (new_ao.rgba.a < -5.5) {
                        // This Splat never registred so we subtract it before we break.
                        // local_n_contrib--;
                        thread_active = false;
                        break;
                    }
                }
            }
        }
        splats_left_to_process -= block_size;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void calc_tet_err(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 TensorView<int32_t> indices,
                 TensorView<float> vertices,
                 TensorView<float> tet_density,
                 TensorView<float> vert_sdf,  // [SDF-MODIFICATION] 全局顶点SDF数组
                 float inv_s,                 // [SDF-MODIFICATION] NeuS inv_s 参数
                 TensorView<float> output_img,
                 TensorView<float> gt,
                 TensorView<float> pixel_err,
                 TensorView<float> ssim_err,
                 TensorView<float> tet_err,
                 TensorView<int32_t> tet_count,
                 TensorView<int32_t> n_contributors,
                 TensorCamera tcam)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    Camera cam = load_tensor_camera(tcam);
    uint32_t tile_idx = cudaBlockIdx().y * cam.grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));
    float this_pixel_err = 0;
    float this_ssim_err = 0;
    float3 this_gt = {0.f, 0.f, 0.f};
    if (is_inside) {

        this_pixel_err = pixel_err[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))];
        this_ssim_err = ssim_err[uint2(uint32_t(pix_coord.y), uint32_t(pix_coord.x))];
        this_gt = {
            gt[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
            gt[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
            gt[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
        };

    }
    kern_calc_tet_err(
        sorted_gauss_idx,
        indices,
        vertices,
        tet_density,
        vert_sdf,      // [SDF-MODIFICATION] 全局顶点SDF数组
        inv_s,         // [SDF-MODIFICATION] NeuS inv_s 参数
        output_img,
        tet_err,
        tet_count,
        n_contributors,
        cam,
        pix_coord,
        tile_idx_start,
        tile_idx_end,
        this_pixel_err,
        this_ssim_err,
        this_gt);
}
