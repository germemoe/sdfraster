import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.camera import Camera
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from delaunay_rasterization.internal.alphablend_tiled_slang import render_constant_color
import time

cmap = plt.get_cmap("jet")

def render(camera: Camera, model, cell_values=None, tile_size=16, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None,
           **kwargs):
    device = model.device
    if ray_jitter is None:
        ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 2), device=device)
    else:
        assert(ray_jitter.shape[0] == camera.image_height)
        assert(ray_jitter.shape[1] == camera.image_width)
        assert(ray_jitter.shape[2] == 2)
    vertices = model.vertices

    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        **camera.to_dict(device)
    )
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices,
        vertices,
        tcam,
        render_grid)
    extras = {}
    sh_reg = 0
    if cell_values is None:
        cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
        if mask.sum() > 0 and model.mask_values:
            shs, values = model.get_cell_values(camera, mask, circumcenter[mask])
            cell_values[mask] = values
        else:
            shs, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter)
        sh_reg = (shs**2).mean()

    # [SDF-MODIFICATION] 查询全局顶点SDF并获取inv_s
    vert_sdf = model.query_vert_sdf().reshape(-1, 1)  # (M, 1) 形状
    inv_s = model.backbone.inv_s  # scalar tensor

    # [DEPTH-OUTPUT] AlphaBlendTiledRender 现在返回 5 个值
    image_rgb, distortion_img, normal_img, depth_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        cell_values,
        render_grid,
        vert_sdf,      # [SDF-MODIFICATION] 全局顶点SDF数组
        inv_s,         # [SDF-MODIFICATION] NeuS inv_s 参数
        tcam,
        ray_jitter)
    alpha = image_rgb.permute(2,0,1)[3, ...]
    total_density = (distortion_img[:, :, 2]**2).clip(min=1e-6)
    distortion_loss = (((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4]) / total_density).clip(min=0)

    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...] * camera.gt_alpha_mask.to(device),
        'alpha': alpha,
        'normal': normal_img.permute(2,0,1),  # [NORMAL-OUTPUT] (3, H, W)
        'depth': depth_img,  # [DEPTH-OUTPUT] (H, W)
        'distortion_loss': distortion_loss.mean(),
        'mask': mask,
        'sh_reg': sh_reg,
        **extras
    }
    return render_pkg

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    lr_init = max(lr_init, 1e-20)
    lr_final = max(lr_final, 1e-20)

    def helper(step):
        if max_steps == 0:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def pad_hw2even(h, w):
    return int(math.ceil(h / 2))*2, int(math.ceil(w / 2))*2

def pad_image2even(im, fnp=np):
    h, w = im.shape[:2]
    nh, nw = pad_hw2even(h, w)
    im_full = fnp.zeros((nh, nw, 3), dtype=im.dtype)
    im_full[:h, :w] = im
    return im_full

class SpikingLR:
    def __init__(self, duration, max_steps, base_function,
                 peak_start, peak_interval, peak_end,
                 peak_lr_init, peak_lr_final):
        self.duration = duration
        self.base_function = base_function
        self.max_steps = max_steps

        self.peak_start = peak_start
        self.peak_interval = peak_interval
        self.peak_end = peak_end

        self.peak_lr_init = peak_lr_init
        self.peak_lr_final = peak_lr_final

    def peak_height_fn(self, i):
        return i / self.max_steps * (self.peak_lr_final - self.peak_lr_init) + self.peak_lr_init
        # return self.peak_lr_init
    
    def peak_fn(self, step, height):
        t = np.clip(step / self.duration, 0, 1)
        log_lerp = np.exp(np.log(max(height, 1e-20)) * (1 - t) + np.log(1e-6) * t)
        return log_lerp
        # return height * math.exp(-step * 6/self.duration + 2/self.duration) / math.exp(2/self.duration)

    def __call__(self, iteration):
        base_f = self.base_function(iteration)
        if iteration < self.peak_start:
            return base_f
        elif iteration > self.peak_end:
            last_peak = iteration - self.peak_end
        else:
            last_peak = (iteration - self.peak_start) % self.peak_interval
        peak_ind = iteration - last_peak
        height = self.peak_height_fn(peak_ind) - self.base_function(peak_ind)
        return base_f + self.peak_fn(last_peak, height)

def render_debug(render_tensor, model, camera, density_multi=1, tile_size=16):
    # [SDF-MODIFICATION] density_multi 参数名保留以兼容，但现在用于SDF缩放

    # Convert to RGB (NxMx3) using the colormap
    _, features = model.get_cell_values(camera)
    tet_grad_color = torch.zeros((features.shape[0], 4), device=features.device)
    if render_tensor.shape[1] == 1:
        tensor_min, tensor_max = render_tensor.min(), torch.quantile(render_tensor, 0.99)
        normalized_tensor = ((render_tensor - tensor_min) / (tensor_max - tensor_min)).clip(0, 1)
        normalized_tensor = torch.as_tensor(
            cmap(normalized_tensor.reshape(-1).cpu().numpy())).float().cuda()
    else:
        normalized_tensor = render_tensor
    tet_grad_color[:, :normalized_tensor.shape[1]] = normalized_tensor
    if render_tensor.shape[1] < 4:
        # [SDF-MODIFICATION] 使用SDF(features[:,0])乘以density_multi作为alpha通道
        # TODO: 对于SDF，可能需要不同的可视化策略 (例如: sigmoid(sdf) 或 abs(sdf))
        tet_grad_color[:, 3] = features[:, 0] * density_multi
        # === [DENSITY-RELATED] 原密度代码 (已注释) ===
        # tet_grad_color[:, 3] = features[:, 0] * density_multi  # density 作为alpha
        # === [DENSITY-RELATED] 原密度代码结束 ===
    render_pkg = render_constant_color(model.indices, model.vertices, None, camera, cell_values=tet_grad_color, tile_size=tile_size)

    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = (image.detach().cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8)

    del render_pkg, render_tensor
    return image