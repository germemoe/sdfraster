import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from delaunay_rasterization.internal.util import recombine_tensors, split_tensors

class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                sorted_tetra_idx, tile_ranges,
                indices, vertices, tet_sdf, render_grid,
                # [SDF-MODIFICATION] NeuS 渲染新增参数
                vert_sdf,  # (M,) 全局顶点SDF
                inv_s_tensor,     # (1,) tensor, exp(s_var) for NeuS alpha (可微)
                tcam, ray_jitter, device="cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # === [DENSITY-RELATED] 原密度代码签名 (已注释) ===
        # indices, vertices, tet_density, render_grid, tcam, ray_jitter
        # === [DENSITY-RELATED] 原密度代码结束 ===
        distortion_img = torch.zeros((render_grid.image_height,
                                  render_grid.image_width, 5),
                                 device=device)
        output_img = torch.zeros((render_grid.image_height,
                                  render_grid.image_width, 4),
                                 device=device)
        n_contributors = torch.zeros((render_grid.image_height,
                                      render_grid.image_width, 1),
                                     dtype=torch.int32, device=device)

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=device)
        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders_interp, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders_interp.keys()}'
        )

        # [SDF-MODIFICATION] 确保 inv_s 是一个 (1,) 的 tensor
        if inv_s_tensor.dim() == 0:
            inv_s_tensor = inv_s_tensor.unsqueeze(0)

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders_interp[(render_grid.tile_height, render_grid.tile_width)]
        st = time.time()
        splat_kernel_with_args = alpha_blend_tile_shader.splat_tiled(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=vertices,
            tet_density=tet_sdf,
            # [SDF-MODIFICATION] NeuS 渲染新增参数传递给 Slang
            vert_sdf=vert_sdf,
            inv_s_tensor=inv_s_tensor,  # [SDF-MODIFICATION] 作为可微tensor传递
            output_img=output_img,
            distortion_img=distortion_img,
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            tcam=tcam,
            ray_jitter=ray_jitter,
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width,
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width,
                      render_grid.grid_height, 1)
        )

        tensors = [
            sorted_tetra_idx, tile_ranges,
            indices, vertices, tet_sdf,
            vert_sdf, inv_s_tensor,  # [SDF-MODIFICATION] 保存 NeuS 参数用于反向传播
            output_img, distortion_img, n_contributors,
            ray_jitter
        ]
        non_tensor_data, tensor_data = split_tensors(tcam)
        ctx.save_for_backward(*tensors, *tensor_data)
        ctx.non_tensor_data = non_tensor_data
        ctx.len_tensors = len(tensors)

        ctx.render_grid = render_grid

        return output_img, distortion_img, tet_alive

    @staticmethod
    def backward(ctx, grad_output_img, grad_distortion_img, grad_vert_alive):
        # [SDF-MODIFICATION] 从保存的张量中恢复参数
        (sorted_tetra_idx, tile_ranges,
         indices, vertices, tet_sdf,
         vert_sdf, inv_s_tensor,  # [SDF-MODIFICATION] NeuS 参数
         output_img, distortion_img, n_contributors,
            ray_jitter) = ctx.saved_tensors[:ctx.len_tensors]
        tcam = recombine_tensors(ctx.non_tensor_data, ctx.saved_tensors[ctx.len_tensors:])
        render_grid = ctx.render_grid

        vertices_grad = torch.zeros_like(vertices)
        tet_sdf_grad = torch.zeros_like(tet_sdf)
        vert_sdf_grad = torch.zeros_like(vert_sdf)  # [SDF-MODIFICATION] 顶点SDF梯度
        inv_s_grad = torch.zeros_like(inv_s_tensor)  # [SDF-MODIFICATION] inv_s 梯度 (可微)

        assert (render_grid.tile_height, render_grid.tile_width) in slang_modules.alpha_blend_shaders_interp, (
            'Alpha Blend Shader was not compiled for this tile'
            f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
            f' {slang_modules.alpha_blend_shaders_interp.keys()}'
        )

        alpha_blend_tile_shader = slang_modules.alpha_blend_shaders_interp[(render_grid.tile_height, render_grid.tile_width)]

        tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=vertices.device)
        st = time.time()
        kernel_with_args = alpha_blend_tile_shader.splat_tiled.bwd(
            sorted_gauss_idx=sorted_tetra_idx,
            tile_ranges=tile_ranges,
            indices=indices,
            vertices=(vertices, vertices_grad),
            tcam=tcam,
            tet_density=(tet_sdf, tet_sdf_grad),
            # [SDF-MODIFICATION] NeuS 参数反向传播 (可微)
            vert_sdf=(vert_sdf, vert_sdf_grad),
            inv_s_tensor=(inv_s_tensor, inv_s_grad),  # [SDF-MODIFICATION] 可微参数
            output_img=(output_img, grad_output_img),
            distortion_img=(distortion_img, grad_distortion_img),
            n_contributors=n_contributors,
            tet_alive=tet_alive,
            ray_jitter=ray_jitter)

        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width,
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width,
                      render_grid.grid_height, 1)
        )

        # [SDF-MODIFICATION] 返回所有梯度
        # 顺序: sorted_tetra_idx, tile_ranges, indices, vertices, tet_sdf, render_grid, vert_sdf, inv_s_tensor, tcam, ray_jitter, device
        # inv_s_tensor 现在是可微的，返回其梯度
        return (None, None, None, vertices_grad, tet_sdf_grad,
                None, vert_sdf_grad, inv_s_grad, None, None, None)
