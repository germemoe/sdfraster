import torch
import math
from data.camera import Camera
from utils import optim
from gdel3d import Del
from torch import nn
from icecream import ic

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere
from utils import topo_utils
from utils.contraction import contract_mean_std

from utils.train_util import get_expon_lr_func, SpikingLR
from utils.graphics_utils import l2_normalize_th
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
from scipy.spatial import  Delaunay, ConvexHull
import open3d as o3d
from utils.model_util import *
from models.base_model import BaseModel
from utils.ingp_util import grid_scale, compute_grid_offsets
from utils.hashgrid import HashEmbedderOptimized
from utils import hashgrid
import tinycudann as tcnn
import time

# [SDF-MODIFICATION] 几何初始化函数，参考Neuralangelo的MLPforNeuralSDF
def geometric_init_sdf_layer(linear: nn.Linear, k_in: int, out_bias: float = 0.5, invert: bool = False):
    """
    Initialize the SDF output layer with geometric initialization.
    This initialization makes the network output approximate a sphere SDF initially.

    Args:
        linear: The linear layer to initialize
        k_in: Input dimension
        out_bias: Initial bias value (controls initial sphere radius)
        invert: If True, invert the SDF (inside becomes outside)
    """
    nn.init.normal_(linear.weight, mean=np.sqrt(np.pi / k_in), std=0.0001)
    nn.init.constant_(linear.bias, -out_bias)
    if invert:
        linear.weight.data *= -1
        linear.bias.data *= -1

torch.set_float32_matmul_precision('high')

def pad_for_tinycudann(x: torch.Tensor, granularity: int):
    """
    Pads the input tensor to ensure its total number of elements is a multiple
    of the tinycudann's required batch size granularity.

    Args:
        x (torch.Tensor): The input tensor to the tinycudann module.
        granularity (int): The BATCH_SIZE_GRANULARITY required by tinycudann.

    Returns:
        tuple: A tuple containing the padded tensor and the number of padded elements.
    """
    num_elements = x.shape[0]
    remainder = num_elements % granularity

    if remainder == 0:
        return x, 0
    else:
        padding_needed = granularity - remainder
        padding_shape = list(x.shape)
        padding_shape[0] = padding_needed
        padding = torch.zeros(padding_shape, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, padding], dim=0)
        
        return x_padded, padding_needed

def gaussian_in_circumsphere(cc: torch.Tensor,       # (T,3)
                             r:  torch.Tensor,       # (T,1)
                             k:  int,
                             trunc_sigma: float = 0.3) -> torch.Tensor:
    """
    Draw `k` 3‑D points from N(cc, (trunc_sigma*r)^2 I), truncated so ‖x‑cc‖≤r.

    Returns: (T,k,3)
    """
    T = cc.shape[0]
    # iid standard normal                                                          (T,k,3)
    x  = torch.randn((T, k, 3), device=cc.device)

    # scale by radius*trunc_sigma
    x  = x * (trunc_sigma * r).unsqueeze(1)

    # rejection‑sampling for the few out‑of‑sphere samples ------------------------
    inside = (x.norm(dim=-1, p=2) <= r.unsqueeze(1)).all(dim=-1)
    while not inside.all():
        # re‑draw only the failed rows (≈ 1 % for σ=0.3)
        mask   = ~inside
        num    = mask.sum()
        x[mask] = torch.randn((num, 3), device=x.device) * (trunc_sigma * r[mask])
        inside  = (x.norm(dim=-1) <= r[mask]).all(dim=-1)

    return cc.unsqueeze(1) + x                 # (T,k,3)

def approx_erf(x):
  """An approximation of erf() that is accurate to within 0.007."""
  return torch.sign(x) * torch.sqrt(1 - torch.exp(-(4 / torch.pi) * x**2))

class iNGPDW(nn.Module):
    def __init__(self,
                 sh_dim=0,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 g_init=1,
                 s_init=1e-4,
                 d_init=0.1,
                 c_init=0.6,
                 density_offset=-4,  # [DENSITY-RELATED] 密度偏移参数 (保留用于兼容性)
                 sdf_bias=0.5,  # [SDF-MODIFICATION] SDF初始偏移，控制初始球体半径
                 sdf_inside_out=False,  # [SDF-MODIFICATION] 是否反转SDF内外
                 # [SDF-MODIFICATION] NeuS s_var 参数 (参考 Neuralangelo)
                 s_var_init=1.4,  # 初始值 (存储的是 log(inv_s)，1.4 对应 inv_s≈4.05，适合扫描数据)
                 s_var_anneal_end=0.1,  # schedule 模式的结束进度
                 s_var_learn=True,  # True: 可学习; False: 按 schedule 衰减
                 ablate_downweighing=False,
                 k_samples=1,
                 trunc_sigma=1,
                 use_tcnn=False,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset  # [DENSITY-RELATED] 保留用于兼容性
        self.sdf_bias = sdf_bias  # [SDF-MODIFICATION] SDF偏移参数
        self.sdf_inside_out = sdf_inside_out  # [SDF-MODIFICATION] SDF内外反转标志
        self.ablate_downweighing = ablate_downweighing
        self.k_samples = k_samples
        self.trunc_sigma = trunc_sigma

        # [SDF-MODIFICATION] NeuS s_var 参数初始化 (参考 Neuralangelo)
        # s_var 存储的是 log(inv_s)，使用时 inv_s = exp(s_var)
        # Φ(sdf) = sigmoid(sdf * inv_s) = sigmoid(sdf * exp(s_var))
        self.s_var_init = s_var_init
        self.s_var_anneal_end = s_var_anneal_end
        self.s_var_learn = s_var_learn
        if s_var_learn:
            # 可学习参数
            self.s_var = nn.Parameter(torch.tensor(s_var_init, dtype=torch.float32))
        else:
            # 不可学习，通过 buffer 存储（会被保存到 checkpoint）
            self.register_buffer('s_var', torch.tensor(s_var_init, dtype=torch.float32))

        self.config = dict(
            per_level_scale=per_level_scale,
            n_levels=L,
            otype="HashGrid",
            n_features_per_level=self.dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )
        if use_tcnn:
            self.encoding = tcnn.Encoding(3, self.config)
            print("Using TCNN")
            self.compile = False
        else:
            print("Using PyTorch iNGP")
            self.encoding = hashgrid.HashEmbedderOptimized(
                [torch.zeros((3)), torch.ones((3))],
                self.L, n_features_per_level=self.dim,
                log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
                finest_resolution=base_resolution*per_level_scale**self.L)
            self.compile = True


        def mk_head(n):
            network = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, n)
            )
            gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
            network.apply(lambda m: init_linear(m, gain))
            return network

        self.network = mk_head(1+12+sh_dim)

        # [SDF-MODIFICATION] 创建SDF预测网络，替代原来的density_net
        # SDF网络使用几何初始化，使初始输出近似为球体SDF
        self.sdf_net = mk_head(1)  # [SDF-MODIFICATION] SDF预测网络头
        self.density_net = self.sdf_net  # [DENSITY-RELATED] 保留别名用于兼容性
        self.color_net     = mk_head(3)
        self.gradient_net  = mk_head(3)
        self.sh_net        = mk_head(sh_dim)

        last = self.network[-1]
        with torch.no_grad():
            last.weight[4:, :].zero_()
            last.bias[4:].zero_()
            # 颜色、梯度、SH网络使用Xavier初始化
            for network, eps in zip(
                [self.gradient_net, self.sh_net, self.color_net],
                [g_init, s_init, c_init]):
                last = network[-1]
                with torch.no_grad():
                    nn.init.uniform_(last.weight.data, a=-eps, b=eps)
                    last.bias.zero_()

            # [SDF-MODIFICATION] SDF网络使用几何初始化
            sdf_last = self.sdf_net[-1]
            geometric_init_sdf_layer(sdf_last, sdf_last.in_features,
                                    out_bias=sdf_bias, invert=sdf_inside_out)


    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        output = self.encoding(x)
        output = output.reshape(-1, self.dim, self.L)
        if not self.ablate_downweighing:
            cr = cr.detach() * self.scale_multi
            n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
            erf_x = safe_div(torch.tensor(1.0, device=x.device),
                            safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
            scaling = approx_erf(erf_x)
            output = output * scaling
        return output


    def forward(self, x, cr):
        # output = self._encode(x, cr)
        if self.k_samples > 1:
            output = self._encode(x, cr)
            output = output.view(-1, self.k_samples, output.shape[-1]).mean(dim=1)
        else:
            output = self._encode(x, cr)

        h = output.reshape(-1, self.L * self.dim).float()

        # [SDF-MODIFICATION] SDF网络前向传播，直接输出SDF值
        sdf = self.sdf_net(h)  # [SDF-MODIFICATION] 输出SDF值 (可正可负)
        # === [DENSITY-RELATED] 原密度代码 (已注释) ===
        # sigma = self.density_net(h)  # [DENSITY-RELATED] 密度网络前向传播
        # density = safe_exp(sigma + self.density_offset)  # [DENSITY-RELATED] 密度激活函数
        # === [DENSITY-RELATED] 原密度代码结束 ===

        rgb = self.color_net(h)
        field_samples = self.gradient_net(h)
        sh  = self.sh_net(h).half()

        rgb = rgb.reshape(-1, 3, 1) + 0.5
        # [SDF-MODIFICATION] SDF直接输出，不需要激活函数
        # 注意：后续渲染代码需要将SDF转换为alpha (使用NeuS公式)
        # grd = torch.tanh(field_samples.reshape(-1, 1, 3)) / math.sqrt(3)
        grd = field_samples.reshape(-1, 1, 3)
        grd = grd / ((grd * grd).sum(dim=-1, keepdim=True) + 1).sqrt()
        # grd = rgb * torch.tanh(field_samples.reshape(-1, 3, 3))  # shape (T, 3, 3)
        # [SDF-MODIFICATION] 返回SDF作为第一个元素
        return sdf, rgb.reshape(-1, 3), grd, sh

    # [SDF-MODIFICATION] NeuS s_var 相关方法
    @property
    def inv_s(self):
        """获取 inv_s = exp(s_var)，用于 NeuS alpha 计算"""
        return self.s_var.exp()

    def update_s_var(self, progress: float):
        """
        更新 s_var 值（仅在 s_var_learn=False 时生效）
        使用指数衰减 schedule，使表面逐渐变得锐利

        Args:
            progress: 训练进度 [0, 1]，例如 iteration / max_iteration
        """
        if not self.s_var_learn:
            # 在 anneal_end 进度之前进行衰减
            if progress < self.s_var_anneal_end:
                # 线性衰减 s_var (log 空间)
                # 从 s_var_init 衰减到更小的值
                anneal_progress = progress / self.s_var_anneal_end
                # s_var 从 init 线性衰减到 init * 0.1
                new_s_var = self.s_var_init * (1 - 0.9 * anneal_progress)
                self.s_var.fill_(new_s_var)

class Model(BaseModel):
    def __init__(self,
                 vertices: torch.Tensor,
                 ext_vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 density_offset=-1,  # [DENSITY-RELATED] Model类的密度偏移参数 (保留用于兼容性)
                 sdf_bias=0.5,  # [SDF-MODIFICATION] SDF初始偏移参数
                 sdf_inside_out=False,  # [SDF-MODIFICATION] SDF内外反转标志
                 current_sh_deg=2,
                 max_sh_deg=2,
                 ablate_circumsphere=False,
                 ablate_gradient=False,
                 **kwargs):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset  # [DENSITY-RELATED] 存储密度偏移 (保留用于兼容性)
        self.sdf_bias = sdf_bias  # [SDF-MODIFICATION] 存储SDF偏移参数
        self.sdf_inside_out = sdf_inside_out  # [SDF-MODIFICATION] 存储SDF内外反转标志
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
            [math.pi/3, math.pi/3],
        ], device=self.device)
        sh_dim = ((1+max_sh_deg)**2-1)*3

        # [SDF-MODIFICATION] iNGPDW接收SDF参数 (sdf_bias, sdf_inside_out) 通过kwargs传递
        module = iNGPDW(sh_dim, sdf_bias=sdf_bias, sdf_inside_out=sdf_inside_out, **kwargs)
        self.compile = module.compile
        if module.compile:
            self.backbone = torch.compile(module).to(self.device)
        else:
            self.backbone = module.to(self.device)


        # backbone = iNGPDW(sh_dim, **kwargs).to(self.device)
        # sample_cv = torch.rand((512, 3)).to(self.device)
        # sample_cr = torch.rand((512, 1)).to(self.device)
        # traced_backbone = torch.jit.trace(backbone, example_inputs=(sample_cv, sample_cr))
        # self.backbone = traced_backbone

        config = self.backbone.config
        offsets, pred_total = compute_grid_offsets(config, 3)
        total = list(self.backbone.encoding.parameters())[0].shape[0]
        # ic(offsets, pred_total, total)
        # assert total == pred_total, f"Pred #params: {pred_total} vs {total}"
        resolution = grid_scale(
            config['n_levels']-1, config['per_level_scale'], config['base_resolution'])
        self.different_size = 0
        self.nominal_offset_size = offsets[-1] - offsets[-2]
        for o1, o2 in zip(offsets[:-1], offsets[1:]):
            if o2 - o1 == self.nominal_offset_size:
                break
            else:
                self.different_size += 1
        self.offsets = offsets

        self.chunk_size = 308576
        self.mask_values = True
        self.frozen = False
        self.linear = False
        self.feature_dim = 7
        self.alpha = 0
        self.ablate_circumsphere = ablate_circumsphere
        self.ablate_gradient = ablate_gradient

        k_samples = self.backbone.k_samples
        # indices = torch.arange(k_samples, dtype=torch.float32) + 0.5
        # r = (indices / k_samples).pow(1.0/3.0) # Shape [k_samples]
        # golden_ratio = (1. + math.sqrt(5.)) / 2.
        # phi = indices * (2. * math.pi / golden_ratio)  # Azimuthal angle
        # y = 1. - (indices * (2. / k_samples))          # y-coordinate (from 1 down to -1)
        # r_surface = torch.sqrt(1. - y*y)
        # x = r_surface * torch.cos(phi)
        # z = r_surface * torch.sin(phi)
        # surface_dirs = torch.stack([x, y, z], dim=-1) # Shape [k_samples, 3]
        # sphere_pattern = surface_dirs * r.unsqueeze(-1) # Shape [k_samples, 3]

        # sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        # qmc_samples_01 = sobol_engine.draw(k_samples) + 1e-6
        # ic(qmc_samples_01)
        # sphere_pattern = torch.special.ndtri(qmc_samples_01)
        # ic(sphere_pattern*0.35)

        s = math.sqrt(1/3)
        fixed_pattern = torch.tensor([
            [ 0.0,  0.0,  0.0],
            [   s,    s,    s],
            [   s,   -s,   -s],
            [  -s,    s,   -s],
            [  -s,   -s,    s]
        ], dtype=torch.float32)
        sphere_pattern = fixed_pattern[:k_samples]
        self.register_buffer('sphere_pattern', sphere_pattern.to(self.device))

        self.register_buffer('ext_vertices', ext_vertices.to(self.device))
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.contracted_vertices = nn.Parameter(vertices.detach())
        self.update_triangulation()

    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_circumcenters(self):
        circumcenter =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return circumcenter

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        tets = vertices[indices[start:end]]
        if circumcenters is None:
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
            radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        if self.ablate_circumsphere:
            circumcenter = tets.mean(dim=1)
            # radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        if self.training:
            circumcenter += self.alpha*torch.randn_like(circumcenter) * radius.reshape(-1, 1)
        if self.backbone.k_samples > 1:
            # eps = torch.randn((circumcenter.shape[0], self.backbone.k_samples, 3),
            #                   device=circumcenter.device)

            eps = self.sphere_pattern.unsqueeze(0)
            if self.training:
                rand_mat = torch.randn(3, 3, device=circumcenter.device)
                q, _ = torch.linalg.qr(rand_mat)
                if torch.det(q) < 0:
                    q[:, 0] = -q[:, 0] # Flip one axis
                eps = torch.matmul(q, eps.transpose(-1, -2)).transpose(-1, -2)

            sampled_cc = circumcenter.reshape(-1, 1, 3) + eps * radius.reshape(-1, 1, 1) * self.backbone.trunc_sigma
            sampled_radius = radius.reshape(-1, 1, 1).expand(-1, self.backbone.k_samples, 1)
            normalized = (sampled_cc.detach() - self.center) / self.scene_scaling
            cv, cr = contract_mean_std(normalized.reshape(-1, 3), sampled_radius.reshape(-1) / self.scene_scaling)
            x = (cv/2 + 1)/2
        else:
            normalized = (circumcenter.detach() - self.center) / self.scene_scaling
            cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
            x = (cv/2 + 1)/2

        output = checkpoint(self.backbone, x, cr, use_reentrant=True)

        # cr = cr.reshape(-1, 1)
        # x, n = pad_for_tinycudann(x, 256)
        # cr, n = pad_for_tinycudann(cr, 256)
        # N = circumcenter.shape[0]
        # output = self.backbone(x, cr.reshape(-1, 1))
        # output = [v[:N] for v in output]
        return circumcenter, *output

    def get_cell_values(self, camera: Camera, mask=None,
                        all_circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        sh_dim = (self.max_sh_deg+1)**2 - 1
        features = torch.empty((indices.shape[0], self.feature_dim), device=self.device)
        shs = torch.empty((indices.shape[0], sh_dim, 3), device=self.device)
        start = 0
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            # [SDF-MODIFICATION] compute_batch_features 返回 sdf
            circumcenters, sdf, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters)
            # === [DENSITY-RELATED] 原密度代码 (已注释) ===
            # circumcenters, density, rgb, grd, sh = self.compute_batch_features(...)
            # === [DENSITY-RELATED] 原密度代码结束 ===
            if self.ablate_gradient:
                grd = torch.zeros_like(grd)
            centroids = vertices[indices[start:end]].mean(dim=1)
            shs[start:end] = sh.reshape(-1, sh_dim, 3)
            # [SDF-MODIFICATION] activate_output 接收 sdf 参数
            # TODO: activate_output 需要修改以处理SDF而非density
            dvrgbs = activate_output(camera.camera_center.to(self.device),
                                     sdf, rgb, grd,  # [SDF-MODIFICATION] sdf 替代 density
                                     sh.reshape(-1, sh_dim, 3),
                                     indices[start:end],
                                     centroids,
                                     vertices.detach(),
                                     self.current_sh_deg, self.max_sh_deg)
            features[start:end] = dvrgbs
        return shs, features

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, max_sh_deg,
                      voxel_size=0.00, **kwargs):
        sdf_sphere_radius_scale = kwargs.pop('sdf_sphere_radius_scale', 1.2)
        init_num = kwargs.pop('init_num', 0)  # 初始点云最小数量
        torch.manual_seed(2)

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()


        # vertices = vertices.reshape(-1, 1, 3).expand(-1, init_repeat, 3)
        # vertices = vertices + torch.randn(*vertices.shape) * dist.reshape(-1, 1, 1).clip(min=0.01)
        # vertices = vertices.reshape(-1, 3)

        # Convert BasicPointCloud to Open3D PointCloud
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())

        # Perform voxel downsampling
        if voxel_size > 0:
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        N = point_cloud.points.shape[0]
        vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # 如果点云数量不足 init_num，通过扰动扩充
        if init_num > 0 and vertices.shape[0] < init_num:
            original_num = vertices.shape[0]
            # 计算需要扩充的倍数
            repeat_times = int(np.ceil(init_num / original_num))
            # 复制点云并添加小扰动
            vertices_list = [vertices]
            for i in range(repeat_times - 1):
                perturbed = vertices + torch.randn_like(vertices) * 1e-2
                vertices_list.append(perturbed)
            vertices = torch.cat(vertices_list, dim=0)
            # 截取到 init_num
            vertices = vertices[:init_num]
            print(f"[Init] Expanded point cloud from {original_num} to {vertices.shape[0]} points")

        # [SDF-MODIFICATION] 参考 svraster 的点云边界计算方式
        # 使用点云中位数作为中心，基于密度启发式计算半径（排除离群点）
        pcd_center = vertices.median(dim=0).values  # 点云中位数作为中心
        pcd_dists = torch.linalg.norm(vertices - pcd_center, dim=1, ord=2)  # 到点云中心的距离

        # 基于密度启发式找半径（参考 svraster 的 main_scene_bound_pcd_heuristic）
        pcd_dists_sorted, _ = pcd_dists.sort()
        pcd_density_rate = 0.1  # 密度阈值比例
        # 计算密度: N(r) / V(r)，其中 V(r) ∝ r^3
        valid_mask = pcd_dists_sorted > 0
        density = torch.zeros_like(pcd_dists_sorted)
        density[valid_mask] = (1 + torch.arange(len(pcd_dists_sorted), device=vertices.device).float()[valid_mask]) / ((2 * pcd_dists_sorted[valid_mask]) ** 3 + 1e-6)

        # 至少覆盖 5% 的点
        begin_idx = round(len(density) * 0.05)
        # 找密度最大的位置
        max_idx = begin_idx + density[begin_idx:].argmax().item()
        # 找密度下降到最大值 * pcd_density_rate 的位置
        target_density = pcd_density_rate * density[max_idx]
        below_target = density[max_idx:] < target_density
        if below_target.any():
            target_idx = max_idx + torch.where(below_target)[0][0].item()
        else:
            target_idx = len(density) - 1

        pcd_radius = pcd_dists_sorted[target_idx]
        new_radius = pcd_radius.cpu().item()

        # 计算 sdf_bias（网络输入空间中的球体半径）
        # 坐标变换流程: world → normalized(÷scaling) → contracted → x=(cv/2+1)/2
        sdf_bias = (pcd_radius * sdf_sphere_radius_scale / scaling / 2).item()
        print(f"[SDF Init] Point cloud center: {pcd_center}, "
              f"pcd_radius (density-based): {pcd_radius:.4f}, "
              f"scene_scaling: {scaling:.4f}, "
              f"sdf_bias (network space): {sdf_bias:.4f}")

        # vertices = sample_uniform_in_sphere(10000, 3, base_radius=0, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()

        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        # v = Del(vertices.shape[0])
        # indices_np, prev = v.compute(vertices.detach().cpu().double())
        # indices_np = indices_np.numpy()
        # indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        # vertices = vertices[indices_np].mean(dim=1)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # within_sphere = sample_uniform_in_sphere(10000, 3, base_radius=new_radius, radius=new_radius, device='cpu') + center.reshape(1, 3).cpu()
        # vertices = torch.cat([vertices, within_sphere], dim=0)

        # 外部球面点（保持原始逻辑）：以相机中心为圆心，半径为点云到相机中心的最大距离
        pcd_scaling = torch.linalg.norm(vertices - center.cpu().reshape(1, 3), dim=1, ord=2).max()
        new_radius = pcd_scaling.cpu().item()
        num_ext = 1000
        ext_vertices = fibonacci_spiral_on_sphere(num_ext, new_radius, device='cpu') + center.reshape(1, 3).cpu()
        num_ext = ext_vertices.shape[0]

        # num_ext = 1000
        # ext_vertices = topo_utils.expand_convex_hull(vertices, 1, device=vertices.device)
        # if ext_vertices.shape[0] > num_ext:
        #     inds = np.random.default_rng().permutation(ext_vertices.shape[0])[:num_ext]
        #     ext_vertices = ext_vertices[inds]
        # else:
        #     num_ext = ext_vertices.shape[0]
        # vertices = torch.cat([vertices, ext_vertices], dim=0)
        # ext_vertices = torch.empty((0, 3))

        model = Model(vertices.cuda(), ext_vertices, center, scaling,
                      max_sh_deg=max_sh_deg, sdf_bias=sdf_bias, **kwargs)
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        print(ckpt.keys())
        vertices = ckpt['contracted_vertices']
        indices = ckpt["indices"]  # shape (N,4)
        del ckpt["indices"]
        if 'empty_indices' in ckpt:
            empty_indices = ckpt['empty_indices']
            del ckpt['empty_indices']
        else:
            empty_indices = torch.empty((0, 4), dtype=indices.dtype, device=indices.device)
        print(f"Loaded {vertices.shape[0]} vertices")
        ext_vertices = ckpt['ext_vertices']
        model = Model(vertices.to(device), ext_vertices, ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        base_min_t = getattr(config, "base_min_t", getattr(config, "min_t", 0.4))
        model.min_t = model.scene_scaling * base_min_t
        model.indices = torch.as_tensor(indices).cuda()
        model.empty_indices = torch.as_tensor(empty_indices).cuda()
        return model

    # [SDF-MODIFICATION] 重命名为 calc_tet_sdf，返回SDF值
    def calc_tet_sdf(self):
        """计算所有四面体的SDF值"""
        sdf_list = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])

            # [SDF-MODIFICATION] 从 compute_batch_features 获取 sdf
            _, sdf, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)

            sdf_list.append(sdf.reshape(-1))
        return torch.cat(sdf_list)

    # === [DENSITY-RELATED] 原密度代码 (已注释) ===
    # def calc_tet_density(self):
    #     """计算所有四面体的密度值"""
    #     densities = []
    #     verts = self.vertices
    #     for start in range(0, self.indices.shape[0], self.chunk_size):
    #         end = min(start + self.chunk_size, self.indices.shape[0])
    #         _, density, _, _, _ = self.compute_batch_features(verts, self.indices, start, end)
    #         densities.append(density.reshape(-1))
    #     return torch.cat(densities)
    # === [DENSITY-RELATED] 原密度代码结束 ===

    # [SDF-MODIFICATION] 保留别名以兼容现有代码调用
    def calc_tet_density(self):
        """[DEPRECATED] 请使用 calc_tet_sdf()，此方法保留用于兼容"""
        return self.calc_tet_sdf()

    # [SDF-MODIFICATION] 直接查询顶点SDF值（方案B：全局顶点SDF）
    def query_vert_sdf(self, points=None):
        """
        直接查询点的SDF值，不使用 downweighing（因为 SDF 只和位置有关）

        Args:
            points: (N, 3) 世界坐标，如果为 None 则查询所有顶点
        Returns:
            sdf: (N,) SDF 值
        """
        if points is None:
            points = self.vertices

        sdf_list = []
        for start in range(0, points.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, points.shape[0])
            pts = points[start:end]

            # 1. 坐标归一化
            normalized = (pts - self.center) / self.scene_scaling

            # 2. 空间收缩 (只需要坐标，不需要 std)
            cv, _ = contract_mean_std(normalized, torch.zeros(pts.shape[0], device=pts.device))
            x = (cv / 2 + 1) / 2

            # 3. Hash Grid 编码 (直接用 encoding，不经过 _encode 避免 downweighing)
            h = self.backbone.encoding(x)
            h = h.reshape(-1, self.backbone.L * self.backbone.dim).float()

            # 4. SDF 网络
            sdf = self.backbone.sdf_net(h)
            sdf_list.append(sdf.reshape(-1))

        return torch.cat(sdf_list)

    def compute_sdf_gradients(self, points=None, mode="numerical", taps=4, eps=1e-3):
        """
        计算 SDF 的梯度和 Hessian 对角元素（用于 Eikonal 和 Curvature Loss）

        Args:
            points: (N, 3) 查询点坐标，如果为 None 则使用所有顶点
            mode: "analytical" 使用 autograd，"numerical" 使用数值差分
            taps: 数值差分的采样点数，4 或 6（仅 mode="numerical" 时有效）
            eps: 数值差分的步长

        Returns:
            sdf: (N,) SDF 值
            gradients: (N, 3) 一阶梯度
            hessian_diag: (N, 3) Hessian 对角元素 [∂²f/∂x², ∂²f/∂y², ∂²f/∂z²]
        """
        if points is None:
            points = self.vertices

        if mode == "analytical":
            return self._compute_gradients_analytical(points)
        elif mode == "numerical":
            if taps == 6:
                return self._compute_gradients_numerical_6(points, eps)
            elif taps == 4:
                return self._compute_gradients_numerical_4(points, eps)
            else:
                raise ValueError(f"taps must be 4 or 6, got {taps}")
        else:
            raise ValueError(f"mode must be 'analytical' or 'numerical', got {mode}")

    def _compute_gradients_analytical(self, points):
        """解析梯度：使用 autograd"""
        requires_grad = points.requires_grad
        with torch.enable_grad():
            points = points.detach().requires_grad_(True)
            sdf = self.query_vert_sdf(points)

            # 一阶梯度
            gradients = torch.autograd.grad(
                sdf.sum(), points, create_graph=self.training
            )[0]

            # 二阶梯度 (Hessian 对角元素)
            if self.training:
                hessian_diag = torch.autograd.grad(
                    gradients.sum(), points, create_graph=True
                )[0]
            else:
                hessian_diag = None
                gradients = gradients.detach()

        points.requires_grad_(requires_grad)
        return sdf.detach() if not self.training else sdf, gradients, hessian_diag

    def _compute_gradients_numerical_6(self, points, eps):
        """数值梯度：6点采样（中心差分）"""
        device = points.device
        dtype = points.dtype

        # 中心点 SDF
        sdf = self.query_vert_sdf(points)  # (N,)

        # 6个偏移方向
        eps_x = torch.tensor([eps, 0., 0.], dtype=dtype, device=device)
        eps_y = torch.tensor([0., eps, 0.], dtype=dtype, device=device)
        eps_z = torch.tensor([0., 0., eps], dtype=dtype, device=device)

        # 计算6个偏移点的 SDF
        sdf_x_pos = self.query_vert_sdf(points + eps_x)
        sdf_x_neg = self.query_vert_sdf(points - eps_x)
        sdf_y_pos = self.query_vert_sdf(points + eps_y)
        sdf_y_neg = self.query_vert_sdf(points - eps_y)
        sdf_z_pos = self.query_vert_sdf(points + eps_z)
        sdf_z_neg = self.query_vert_sdf(points - eps_z)

        # 一阶梯度（中心差分）
        gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
        gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
        gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
        gradients = torch.stack([gradient_x, gradient_y, gradient_z], dim=-1)  # (N, 3)

        # 二阶梯度（Hessian 对角元素）
        hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)
        hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)
        hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)
        hessian_diag = torch.stack([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # (N, 3)

        return sdf, gradients, hessian_diag

    def _compute_gradients_numerical_4(self, points, eps):
        """数值梯度：4点采样（四面体采样）"""
        device = points.device
        dtype = points.dtype

        # 中心点 SDF
        sdf = self.query_vert_sdf(points)  # (N,)

        # 四面体的4个顶点方向
        eps_adj = eps / math.sqrt(3)  # 调整步长使得偏移距离为 eps
        k1 = torch.tensor([1., -1., -1.], dtype=dtype, device=device) * eps_adj
        k2 = torch.tensor([-1., -1., 1.], dtype=dtype, device=device) * eps_adj
        k3 = torch.tensor([-1., 1., -1.], dtype=dtype, device=device) * eps_adj
        k4 = torch.tensor([1., 1., 1.], dtype=dtype, device=device) * eps_adj

        # 计算4个偏移点的 SDF
        sdf1 = self.query_vert_sdf(points + k1)
        sdf2 = self.query_vert_sdf(points + k2)
        sdf3 = self.query_vert_sdf(points + k3)
        sdf4 = self.query_vert_sdf(points + k4)

        # 一阶梯度
        # gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4 * eps_adj)
        k1_norm = torch.tensor([1., -1., -1.], dtype=dtype, device=device)
        k2_norm = torch.tensor([-1., -1., 1.], dtype=dtype, device=device)
        k3_norm = torch.tensor([-1., 1., -1.], dtype=dtype, device=device)
        k4_norm = torch.tensor([1., 1., 1.], dtype=dtype, device=device)

        gradients = (k1_norm * sdf1.unsqueeze(-1) +
                     k2_norm * sdf2.unsqueeze(-1) +
                     k3_norm * sdf3.unsqueeze(-1) +
                     k4_norm * sdf4.unsqueeze(-1)) / (4.0 * eps_adj)  # (N, 3)

        # 二阶梯度（Hessian 对角元素的近似）
        # 4点采样只能得到 Laplacian 的近似，假设各向同性
        hessian_trace = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / (eps_adj ** 2)
        hessian_diag = torch.stack([hessian_trace, hessian_trace, hessian_trace], dim=-1) / 3.0  # (N, 3)

        return sdf, gradients, hessian_diag

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        cs, sdf_list, rs, gs, ss = [], [], [], [], []  # [SDF-MODIFICATION] ds -> sdf_list
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            # [SDF-MODIFICATION] compute_features 返回 sdf
            circumcenters, sdf, rgb, grd, sh = self.compute_batch_features(vertices, indices, start, end)
            # === [DENSITY-RELATED] 原密度代码 (已注释) ===
            # circumcenters, density, rgb, grd, sh = self.compute_batch_features(...)
            # ds.append(density)
            # === [DENSITY-RELATED] 原密度代码结束 ===
            tets = vertices[indices[start:end]]
            cs.append(circumcenters)
            sdf_list.append(sdf)  # [SDF-MODIFICATION] 收集sdf
            ss.append(sh)
            if offset:
                base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)
                # rs.append(base_color_v0_raw)
                rs.append(rgb)
                gs.append(normed_grd)
            else:
                rs.append(rgb)
                gs.append(grd)
        cs = torch.cat(cs, dim=0)
        sdf_all = torch.cat(sdf_list, dim=0)  # [SDF-MODIFICATION] ds -> sdf_all
        rs = torch.cat(rs, dim=0)
        gs = torch.cat(gs, dim=0)
        ss = torch.cat(ss, dim=0)
        return cs, sdf_all, rs, gs, ss  # [SDF-MODIFICATION] 返回 sdf_all

    @property
    def vertices(self):
        verts = self.contracted_vertices
        return torch.cat([verts, self.ext_vertices])

    @property
    def active_vertices(self):
        """获取活跃顶点（属于未被裁剪四面体的顶点）"""
        active_indices = self.indices.unique()
        return self.vertices[active_indices]

    @property
    def active_vertex_indices(self):
        """获取活跃顶点的索引"""
        return self.indices.unique()

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg+1)

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, sdf_eps=0.0, alpha_threshold=0.0, **kwargs):
        """
        更新 Delaunay 三角剖分并裁剪远离表面的四面体

        Args:
            high_precision: 是否使用高精度 Delaunay
            sdf_eps: CDF 贡献阈值 ε，保留 CDF 贡献 > ε 的四面体
                     例如 ε=0.01 表示保留贡献 > 1% 的四面体
                     内部计算 k = ln((1-ε)/ε)，SDF 阈值 = k / inv_s
                     设为 0 禁用此裁剪
            alpha_threshold: NeuS alpha 阈值，如果四面体最大可能 alpha < threshold，则裁剪
                            设为 0 禁用此裁剪
        """
        torch.cuda.empty_cache()
        verts = self.vertices
        if high_precision:
            indices_np = Delaunay(verts.detach().cpu().numpy()).simplices.astype(np.int32)
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.numpy()
            indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
            del prev

        # Ensure volume is positive
        indices = torch.as_tensor(indices_np).cuda()
        vols = topo_utils.tet_volumes(verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        # [SDF-MODIFICATION] 使用 NeuS 风格的裁剪逻辑
        self.indices = indices
        if sdf_eps > 0 or alpha_threshold > 0:
            # 获取所有顶点的 SDF
            vert_sdf = self.query_vert_sdf()  # (M,)

            # 获取每个四面体 4 个顶点的 SDF
            tet_vert_sdf = vert_sdf[indices.long()]  # (N, 4)

            # 条件1: 基于 CDF 贡献阈值 ε 计算动态 SDF 阈值
            # k = ln((1-ε)/ε)，当 |sdf| * inv_s > k 时，CDF 贡献 < ε
            # 真正的 SDF 阈值 = k / inv_s
            if sdf_eps > 0:
                inv_s = self.backbone.inv_s
                k = math.log((1 - sdf_eps) / sdf_eps)  # 从 ε 计算 k
                sdf_threshold = k / inv_s  # 动态计算真正的 SDF 阈值
                tet_min_abs_sdf = tet_vert_sdf.abs().min(dim=1)[0]  # (N,)
                sdf_mask = tet_min_abs_sdf < sdf_threshold  # 保留靠近表面的
            else:
                sdf_mask = torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device)

            # 条件2: NeuS alpha 阈值
            if alpha_threshold > 0:
                tet_alpha = self.calc_tet_alpha()  # 使用新的 NeuS alpha 计算
                alpha_mask = tet_alpha > alpha_threshold  # 保留有贡献的
            else:
                alpha_mask = torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device)

            # 两个条件取并集 (只要满足其一就保留)
            mask = sdf_mask | alpha_mask
            self.empty_indices = self.indices[~mask]
            self.indices = self.indices[mask]
            self.mask = mask
        else:
            self.empty_indices = torch.empty((0, 4), dtype=self.indices.dtype, device='cuda')

        torch.cuda.empty_cache()

    def __len__(self):
        return self.vertices.shape[0]
        

    def compute_weight_decay(self):
        if self.compile:
            return sum([(embed.weight**2).mean() for embed in self.backbone.encoding.embeddings])
        else:
            param = list(self.backbone.encoding.parameters())[0]
            weight_decay = 0
            ind = 0
            for i in range(self.different_size):
                o = self.offsets[i+1] - self.offsets[i]
                weight_decay = weight_decay + (param[ind:self.offsets[i+1]]**2).mean()
                ind += o
            weight_decay = weight_decay + (param[ind:].reshape(-1, self.nominal_offset_size)**2).mean(dim=1).sum()
            return weight_decay

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 encoding_lr: float=1e-2,
                 final_encoding_lr: float=1e-2,
                 network_lr: float=1e-3,
                 final_network_lr: float=1e-3,
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_multi: float=0.01,
                 lr_delay: int = 500,
                 freeze_start: int = 10000,
                 vert_lr_delay: int = 500,

                 spike_duration: int = 20,
                 densify_interval: int = 500,
                 densify_end: int = 15000,
                 midpoint: int = 2000,

                 percent_alpha: float = 0.02,

                 **kwargs):

        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.999], eps=1e-15)

        # [SDF-MODIFICATION] 构建优化器参数组（根据 s_var_learn 决定是否包含 s_var）
        net_param_groups = [
            # [SDF-MODIFICATION] sdf_net 优化器配置 (通过density_net别名访问)
            # [DENSITY-RELATED] 保留name="density"以兼容现有代码
            {"params": model.backbone.sdf_net.parameters(),   "lr": network_lr,  "name": "sdf"},
            {"params": model.backbone.color_net.parameters(),     "lr": network_lr,    "name": "color"},
            # {"params": model.backbone.density_color_net.parameters(),     "lr": network_lr,    "name": "color"},
            {"params": model.backbone.gradient_net.parameters(),  "lr": network_lr, "name": "gradient"},
            {"params": model.backbone.sh_net.parameters(),        "lr": network_lr,       "name": "sh"},
        ]
        # [SDF-MODIFICATION] 如果 s_var 是可学习参数，添加到优化器
        if model.backbone.s_var_learn:
            net_param_groups.append({"params": [model.backbone.s_var], "lr": network_lr, "name": "s_var"})

        self.net_optim = optim.CustomAdam(net_param_groups, ignore_param_list=[], betas=[0.9, 0.999], eps=1e-15)
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None

        self.alpha_sched = get_expon_lr_func(lr_init=percent_alpha*float(model.scene_scaling.cpu()),
                                                lr_final=1e-20,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=0,
                                                max_steps=freeze_start//3)

        base_net_scheduler = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=freeze_start)

        self.net_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_net_scheduler,
            midpoint, densify_interval, densify_end,
            network_lr, network_lr)
            # network_lr, final_network_lr)

        base_encoder_scheduler = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=freeze_start)

        self.encoder_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_encoder_scheduler,
            midpoint, densify_interval, densify_end,
            # encoding_lr, encoding_lr)
            encoding_lr, encoding_lr)

        self.vertices_lr = self.vert_lr_multi*vertices_lr
        self.final_vertices_lr = self.vert_lr_multi*final_vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertices_lr,
                                                lr_final=self.final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=freeze_start,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = base_vertex_scheduler
        self.vertex_scheduler_args = SpikingLR(
            spike_duration, freeze_start, base_vertex_scheduler,
            midpoint, densify_interval, densify_end,
            # self.vertices_lr, self.final_vertices_lr)
            self.vertices_lr, self.vertices_lr)
        self.iteration = 0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        self.model.alpha = self.alpha_sched(iteration)
        for param_group in self.net_optim.param_groups:
            lr = self.net_scheduler_args(iteration)
            param_group['lr'] = lr
        for param_group in self.optim.param_groups:
            if param_group["name"] == "encoding":
                lr = self.encoder_scheduler_args(iteration)
                param_group['lr'] = lr
        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "contracted_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertices_lr = lr
                param_group['lr'] = lr

    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        keep_mask = keep_mask[:self.model.contracted_vertices.shape[0]]
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(keep_mask)['contracted_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, split_point, **kwargs):
        self.add_points(split_point)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad(set_to_none=True)
        self.net_optim.zero_grad(set_to_none=True)

    @property
    def sh_optim(self):
        return None

    def regularizer(self, render_pkg, lambda_weight_decay, **kwargs):
        # weight_decay = self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])
        weight_decay = lambda_weight_decay * self.model.compute_weight_decay()

        # if self.lambda_density > 0 or self.lambda_tv > 0:
        #     density = self.model.calc_tet_density()
        #     density_loss = (self.model.calc_tet_area().detach() * density).sum()
        #     if self.lambda_tv > 0:
        #         diff  = density[self.pairs[:,0]] - density[self.pairs[:,1]]
        #         tv_loss  = (self.face_area * diff.abs())
        #         tv_loss  = tv_loss.sum() / self.face_area.sum()
        #     else:
        #         tv_loss = 0
        # else:
        #     density_loss = 0
        #     tv_loss = 0

        return weight_decay# + self.lambda_tv * tv_loss + self.lambda_density * density_loss

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)

    # [SDF-MODIFICATION] 基于SDF差异的顶点裁剪
    def prune(self, diff_threshold, **kwargs):
        if diff_threshold <= 0:
            return
        # [SDF-MODIFICATION] 计算四面体SDF
        tet_sdf = self.model.calc_tet_sdf()
        # === [DENSITY-RELATED] 原密度代码 (已注释) ===
        # density = self.model.calc_tet_density()
        # diff = density[self.pairs[:,0]] - density[self.pairs[:,1]]
        # === [DENSITY-RELATED] 原密度代码结束 ===
        # [SDF-MODIFICATION] 计算相邻四面体SDF差异
        diff = tet_sdf[self.pairs[:,0]] - tet_sdf[self.pairs[:,1]]
        tet_diff = (self.face_area * diff.abs()).reshape(-1)

        indices = self.model.indices.long()
        device = indices.device
        vert_diff = torch.zeros((self.model.vertices.shape[0],), device=device)

        reduce_type = "amax"
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 0], src=tet_diff, reduce=reduce_type)
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 1], src=tet_diff, reduce=reduce_type)
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 2], src=tet_diff, reduce=reduce_type)
        vert_diff.scatter_reduce_(dim=0, index=indices[..., 3], src=tet_diff, reduce=reduce_type)
        keep_mask = vert_diff > diff_threshold
        print(f"Pruned {(~keep_mask).sum()} points. VD: {vert_diff.mean()} TD: {tet_diff.mean()}")
        self.remove_points(keep_mask)
