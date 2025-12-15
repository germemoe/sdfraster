import torch
from delaunay_rasterization.internal.render_grid import RenderGrid
import delaunay_rasterization.internal.slang.slang_modules as slang_modules
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from icecream import ic
import time
from data.camera import Camera
from utils.ssim import ssim
import torch.nn.functional as F

# --- quick, fully-differentiable blur ---------------------------------------
def gaussian_blur(img: torch.Tensor,
                  kernel_size: int = 5,
                  sigma: float = 1.5) -> torch.Tensor:
    """
    img : (C,H,W) in [0,1]
    Returns the same-shaped tensor blurred with a depth-wise Gaussian.
    """
    # build 1-D Gaussian
    coords  = torch.arange(kernel_size, device=img.device) - kernel_size // 2
    g1d     = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d     = g1d / g1d.sum()

    # outer product → (k,k) → depth-wise conv kernel (C,1,k,k)
    g2d     = (g1d[:, None] * g1d[None, :]).to(img.dtype)
    kernel  = g2d.expand(img.shape[0], 1, kernel_size, kernel_size).contiguous()

    # depth-wise convolution (groups=C)
    pad = kernel_size // 2
    return F.conv2d(img.unsqueeze(0), kernel, padding=pad,
                    groups=img.shape[0]).squeeze(0)

def render_err(gt_image, camera: Camera, model, tile_size=16, min_t=0.1, **kwargs):
    device = model.device
    indices = model.indices
    vertices = model.vertices
    torch.cuda.synchronize()
    st = time.time()
    device = indices.device
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
    st = time.time()
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        indices, vertices, tcam, render_grid)
   
    # torch.cuda.synchronize()
    # ic("vt", time.time()-st)
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        vs_tetra.retain_grad()
    except:
        pass
    cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
    _, cell_values[mask] = model.get_cell_values(camera, mask)
    # vertex_color, cell_values = model.get_cell_values(camera)

    # torch.cuda.synchronize()
    # st = time.time()
    tet_vertices = vertices[indices]
    distortion_img = torch.zeros((render_grid.image_height, 
                                render_grid.image_width, 4), 
                                device=device)
    output_img = torch.zeros((render_grid.image_height, 
                                render_grid.image_width, 4), 
                                device=device)
    n_contributors = torch.zeros((render_grid.image_height, 
                                    render_grid.image_width, 1),
                                    dtype=torch.int32, device=device)
    tet_alive = torch.zeros((indices.shape[0]), dtype=bool, device=device)
    ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 2), device=device)

    mod = slang_modules.alpha_blend_shaders_interp
    assert (render_grid.tile_height, render_grid.tile_width) in mod, (
        'Alpha Blend Shader was not compiled for this tile'
        f' {render_grid.tile_height}x{render_grid.tile_width} configuration, available configurations:'
        f' {mod.keys()}'
    )

    # [SDF-MODIFICATION] 查询全局顶点SDF并获取inv_s
    vert_sdf = model.query_vert_sdf().reshape(-1, 1)  # (M, 1) 形状
    inv_s = model.backbone.inv_s  # scalar tensor
    # [SDF-MODIFICATION] 确保 inv_s 是一个 (1,) 的 tensor
    inv_s_tensor = inv_s.unsqueeze(0) if inv_s.dim() == 0 else inv_s

    shader = mod[(render_grid.tile_height, render_grid.tile_width)]
    st = time.time()
    # [SDF-MODIFICATION] splat_tiled 参数
    splat_args = dict(
        sorted_gauss_idx=sorted_tetra_idx,
        tile_ranges=tile_ranges,
        indices=indices,
        vertices=vertices,
        tet_density=cell_values,
        vert_sdf=vert_sdf,          # [SDF-MODIFICATION] 全局顶点SDF数组
        inv_s_tensor=inv_s_tensor,  # [SDF-MODIFICATION] NeuS inv_s 参数 (可微tensor)
        output_img=output_img,
        n_contributors=n_contributors,
        tcam=tcam,
    )

    splat_kernel_with_args = shader.splat_tiled(
        **splat_args,
        distortion_img=distortion_img,
        tet_alive=tet_alive,
        ray_jitter=ray_jitter,
    )
    splat_kernel_with_args.launchRaw(
        blockSize=(render_grid.tile_width,
                    render_grid.tile_height, 1),
        gridSize=(render_grid.grid_width,
                    render_grid.grid_height, 1)
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    render_img = output_img.permute(2,0,1)[:3, ...].clip(min=0, max=1)
    l1_err = ((render_img - gt_image)).mean(dim=0)
    ssim_err = (1-ssim(render_img, gt_image).mean(dim=0)).clip(min=0, max=1)
    pixel_err = (l1_err).contiguous()

    mask = camera.gt_alpha_mask.cuda()
    gt_img = (mask*gt_image).permute(1, 2, 0)

    pixel_err *= mask[0]
    ssim_err *= mask[0]
    render_img *= mask

    assert(pixel_err.shape[0] == render_grid.image_height)
    assert(pixel_err.shape[1] == render_grid.image_width)
    assert(ssim_err.shape[0] == render_grid.image_height)
    assert(ssim_err.shape[1] == render_grid.image_width)
    assert(gt_img.shape[0] == render_grid.image_height)
    assert(gt_img.shape[1] == render_grid.image_width)

    tet_err = torch.zeros((tet_vertices.shape[0], 16), dtype=torch.float, device=device)
    tet_count = torch.zeros((tet_vertices.shape[0], 2), dtype=torch.int32, device=device)

    # [SDF-MODIFICATION] calc_tet_err 使用 float inv_s，需要单独传递参数
    inv_s_value = inv_s.item() if inv_s.dim() == 0 else inv_s[0].item()
    shader.calc_tet_err(
        sorted_gauss_idx=sorted_tetra_idx,
        tile_ranges=tile_ranges,
        indices=indices,
        vertices=vertices,
        tet_density=cell_values,
        vert_sdf=vert_sdf,
        inv_s=inv_s_value,  # [SDF-MODIFICATION] calc_tet_err 使用 float
        output_img=output_img,
        n_contributors=n_contributors,
        tcam=tcam,
        pixel_err=pixel_err.contiguous(),
        ssim_err=ssim_err.contiguous(),
        gt=gt_img.contiguous(),
        tet_err=tet_err,
        tet_count=tet_count,
    ).launchRaw(
        blockSize=(render_grid.tile_width,
                    render_grid.tile_height, 1),
        gridSize=(render_grid.grid_width,
                    render_grid.grid_height, 1)
    )

    torch.cuda.synchronize()
    return tet_err, dict(
        tet_count = tet_count,
        pixel_err = pixel_err,
        ssim_err = ssim_err,
        render_img = render_img,
        cell_values = cell_values,
    )
