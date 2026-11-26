import torch
import math
from data.camera import Camera
from utils import optim
from torch import nn
from icecream import ic

from utils.topo_utils import calculate_circumcenters_torch, fibonacci_spiral_on_sphere, calc_barycentric, sample_uniform_in_sphere
from utils import topo_utils
from utils.contraction import contract_mean_std

from utils.train_util import get_expon_lr_func, SpikingLR
from utils.training_clocks import resolve_optimizer_lr
from utils.graphics_utils import l2_normalize_th
from torch.utils.checkpoint import checkpoint
from pathlib import Path
import numpy as np
from utils.args import Args
from utils.encoding_precision import validate_encoding_precision
from utils.matmul_precision import configure_float32_matmul_precision
from scipy.spatial import ConvexHull
from utils.triangulation_backend import TriangulationBackend
import open3d as o3d
from utils.model_util import *
from models.base_model import BaseModel
from utils.ingp_util import grid_scale, compute_grid_offsets
from utils.hashgrid import HashEmbedderOptimized
from utils import hashgrid
import tinycudann as tcnn
import time

torch.set_float32_matmul_precision('high')

# [GEOMETRIC-INIT] 几何初始化函数，参考 Neuralangelo 的 MLPforNeuralSDF
def geometric_init_sdf_layer(linear: nn.Linear, k_in: int, out_bias: float = 0.5, invert: bool = False):
    """
    Initialize the SDF output layer with geometric initialization.
    This initialization makes the network output approximate a sphere SDF initially.

    Args:
        linear: The linear layer to initialize
        k_in: Input dimension
        out_bias: Initial bias value (controls initial sphere radius)
        invert: If True, invert the SDF (inside becomes outside) - only inverts weight, not bias
    """
    nn.init.normal_(linear.weight, mean=np.sqrt(np.pi / k_in), std=0.0001)
    nn.init.constant_(linear.bias, -out_bias)
    if invert:
        linear.weight.data *= -1

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
                 # [GEOMETRIC-INIT] Neuralangelo 风格的几何初始化参数
                 use_geometric_init=False,  # True: xyz输入+几何初始化, False: 仅hash特征
                 sdf_bias=0.5,              # 几何初始化的 SDF 偏移（控制初始球体半径）
                 sdf_inside_out=False,      # 是否反转 SDF 内外
                 # [SDF-MODIFICATION] NeuS s_var 参数 (参考 Neuralangelo)
                 s_var_init=1.4,  # 初始值 (存储的是 log(inv_s)，1.4 对应 inv_s≈4.05，适合扫描数据)
                 s_var_anneal_end=0.1,  # schedule 模式的结束进度
                 s_var_learn=True,  # True: 可学习; False: 按 schedule 衰减
                 ablate_downweighing=False,
                 k_samples=1,
                 trunc_sigma=1,
                 use_tcnn=False,
                 # Coarse-to-Fine 参数 (参考 Neuralangelo)
                 coarse2fine_enabled=False,
                 coarse2fine_init_level=4,
                 coarse2fine_step=5000,
                 encoding_precision="float16",  # TCNN only; PyTorch encoding stays unchanged
                 float32_matmul_precision="high",  # Process-global FP32 GEMM math mode
                 **kwargs):
        super().__init__()
        self.encoding_precision = validate_encoding_precision(encoding_precision)
        self.float32_matmul_precision = configure_float32_matmul_precision(
            float32_matmul_precision)["float32_matmul_precision"]
        self.encoding_backend = "tcnn" if use_tcnn else "pytorch"
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.density_offset = density_offset  # [DENSITY-RELATED] 保留用于兼容性
        self.ablate_downweighing = ablate_downweighing
        self.k_samples = k_samples
        self.trunc_sigma = trunc_sigma

        # [GEOMETRIC-INIT] 保存几何初始化参数
        self.use_geometric_init = use_geometric_init
        self.sdf_bias = sdf_bias
        self.sdf_inside_out = sdf_inside_out

        # Coarse-to-Fine 参数 (参考 Neuralangelo)
        self.coarse2fine_enabled = coarse2fine_enabled
        self.coarse2fine_init_level = min(coarse2fine_init_level, L)
        self.coarse2fine_step = coarse2fine_step
        self.active_levels = self.coarse2fine_init_level if coarse2fine_enabled else L

        # 计算每一层的分辨率 (用于 eps 计算)
        self.resolutions = []
        for lv in range(L):
            res = int(np.floor(base_resolution * (per_level_scale ** lv))) + 1
            self.resolutions.append(res)

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
            encoding_dtype = {"float16": torch.float16, "float32": torch.float32}[self.encoding_precision]
            self.encoding = tcnn.Encoding(3, self.config, dtype=encoding_dtype)
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


        def mk_head(n, input_dim=None):
            """创建 MLP 头部网络
            Args:
                n: 输出维度
                input_dim: 输入维度，默认使用 encoding.n_output_dims
            """
            in_dim = input_dim if input_dim is not None else self.encoding.n_output_dims
            network = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SELU(inplace=True),
                nn.Linear(hidden_dim, n)
            )
            gain = nn.init.calculate_gain('relu')  # for example, if using ReLU activations
            network.apply(lambda m: init_linear(m, gain))
            return network

        def mk_sdf_net_geometric(input_dim, out_bias, inside_out=False):
            """[GEOMETRIC-INIT] 创建 SDF 网络 (对齐 Neuralangelo MLPforNeuralSDF)

            使用 Softplus(β=100) 激活函数，与 Neuralangelo 配置一致。
            结构：1 隐藏层
            - 第一层：weight ~ N(0, sqrt(2/k_out)), bias=0, weight[:, 3:]=0 (hash部分)
            - 输出层：weight ~ N(sqrt(π/k_in), 0.0001), bias=-out_bias
            """
            network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Softplus(beta=100),  # 对齐 Neuralangelo
                nn.Linear(hidden_dim, 1)
            )

            # Neuralangelo 风格几何初始化
            with torch.no_grad():
                linear_layers = [m for m in network if isinstance(m, nn.Linear)]

                # 第一层：先全部初始化，再把 hash 部分置零
                nn.init.constant_(linear_layers[0].bias, 0.0)
                nn.init.normal_(linear_layers[0].weight, 0.0, np.sqrt(2 / hidden_dim))
                if input_dim > 3:
                    nn.init.constant_(linear_layers[0].weight[:, 3:], 0.0)

                # 输出层：球体 SDF 初始化
                nn.init.normal_(linear_layers[-1].weight, mean=np.sqrt(np.pi / hidden_dim), std=0.0001)
                nn.init.constant_(linear_layers[-1].bias, -out_bias)
                if inside_out:
                    linear_layers[-1].weight.data *= -1

            return network

        self.network = mk_head(1+12+sh_dim)

        # [GEOMETRIC-INIT] SDF 网络：根据 use_geometric_init 选择
        if use_geometric_init:
            # Neuralangelo 风格：xyz + hash 特征输入，几何初始化
            sdf_input_dim = 3 + self.encoding.n_output_dims
            self.sdf_net = mk_sdf_net_geometric(sdf_input_dim, sdf_bias, sdf_inside_out)
            print(f"[GEOMETRIC-INIT] SDF net: xyz+hash input ({sdf_input_dim}D), Softplus, geometric init")
        else:
            # 默认：仅 hash 特征输入
            self.sdf_net = mk_head(1)
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


    @torch.no_grad()
    def _get_c2f_mask(self, h: torch.Tensor):
        """
        Coarse-to-Fine mask（和 Neuralangelo 完全对齐）

        Args:
            h: (N, L * dim) hash 特征
        Returns:
            mask: (N, L * dim) mask，前 active_levels * dim 维为 1，其余为 0
        """
        mask = torch.zeros_like(h)
        mask[..., :(self.active_levels * self.dim)] = 1.0
        return mask

    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        output = self.encoding(x)

        # Coarse-to-Fine mask（和 Neuralangelo 对齐）
        if self.coarse2fine_enabled and self.active_levels < self.L:
            output = output * self._get_c2f_mask(output)

        output = output.reshape(-1, self.dim, self.L)

        if not self.ablate_downweighing:
            cr = cr.detach() * self.scale_multi
            n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
            erf_x = safe_div(torch.tensor(1.0, device=x.device),
                            safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
            scaling = approx_erf(erf_x)
            output = output * scaling
        return output

    def set_active_levels(self, iteration):
        """更新 coarse2fine 的激活层数 (参考 Neuralangelo)"""
        if not self.coarse2fine_enabled:
            return
        # 每 coarse2fine_step 迭代增加一层
        new_levels = self.coarse2fine_init_level + iteration // self.coarse2fine_step
        self.active_levels = min(new_levels, self.L)

    def get_normal_epsilon(self):
        """获取当前的数值梯度步长 (参考 Neuralangelo set_normal_epsilon)

        Neuralangelo: eps = 1 / resolution_of_current_active_level
        这样 coarse2fine 时，eps 从大到小，与分辨率匹配
        """
        if self.coarse2fine_enabled:
            # 使用当前激活层的分辨率
            epsilon_res = self.resolutions[self.active_levels - 1]
        else:
            # 使用最细分辨率
            epsilon_res = self.resolutions[-1]
        return 1.0 / epsilon_res

    def get_curvature_weight_decay(
        self,
        iteration: int,
        warmup_end: int = 5000,
        decay_end: int = 30000,
        decay_enabled: bool = True
    ):
        """获取 curvature loss 的权重因子 (参考 Neuralangelo)

        三种模式：
        1. C2F 开启: Neuralangelo 逻辑（warmup + 基于 anneal_levels 衰减）
        2. C2F 关闭 + decay_enabled: warmup 后线性衰减到 final_weight
        3. C2F 关闭 + not decay_enabled: 固定权重 1.0（无 warmup，无 decay）

        最终权重参考 Neuralangelo: 1 / (growth_rate^(L-1))

        Args:
            iteration: 当前迭代数
            warmup_end: warmup 结束的迭代数
            decay_end: 线性衰减结束的迭代数（仅非 C2F 模式）
            decay_enabled: 是否启用衰减（仅非 C2F 模式）

        Returns:
            权重因子
        """
        # 最终权重: 参考 Neuralangelo，1 / (growth_rate^(L-1))
        final_weight = 1.0 / (self.per_level_scale ** (self.L - 1))

        # 模式 1: C2F 开启，使用 Neuralangelo 逻辑
        if self.coarse2fine_enabled:
            if warmup_end > 0 and iteration <= warmup_end:
                return iteration / warmup_end
            else:
                if self.active_levels > 1:
                    decay_factor = self.per_level_scale ** (self.active_levels - 1)
                    return 1.0 / decay_factor
                else:
                    return 1.0

        # 模式 3: 不启用衰减，固定权重
        if not decay_enabled:
            return 1.0

        # 模式 2: 启用衰减，warmup + 线性衰减到 final_weight
        if warmup_end > 0 and iteration <= warmup_end:
            # Warmup: 0 → 1
            return iteration / warmup_end
        elif decay_end > warmup_end and iteration <= decay_end:
            # 线性衰减: 1 → final_weight
            progress = (iteration - warmup_end) / (decay_end - warmup_end)
            return 1.0 - progress * (1.0 - final_weight)
        else:
            return final_weight


    def forward(self, x, cr, compute_sdf=True):
        # x: hash grid 输入坐标 (N, 3)，范围 [0, 1]
        if self.k_samples > 1:
            output = self._encode(x, cr)
            output = output.view(-1, self.k_samples, output.shape[-1]).mean(dim=1)
            xyz_input = x.view(-1, self.k_samples, 3).mean(dim=1)
        else:
            output = self._encode(x, cr)
            xyz_input = x  # [GEOMETRIC-INIT] 直接用 hash 输入坐标，范围 [0, 1]

        h = output.reshape(-1, self.L * self.dim).float()

        # SDF网络前向传播
        # 渲染时 compute_sdf=False（使用 query_vert_sdf 的顶点 SDF），跳过计算节省开销
        if compute_sdf:
            if self.use_geometric_init:
                # [GEOMETRIC-INIT] xyz + hash 特征输入
                # xyz 转换到 [-1, 1] 范围（与旧版本一致）
                xyz_centered = xyz_input * 2 - 1
                sdf_input = torch.cat([xyz_centered, h], dim=-1)
                sdf = self.sdf_net(sdf_input)
            else:
                # 仅 hash 特征输入
                sdf = self.sdf_net(h)
        else:
            sdf = torch.zeros((h.shape[0], 1), device=h.device, dtype=h.dtype)

        rgb = self.color_net(h)
        field_samples = self.gradient_net(h)
        sh  = self.sh_net(h).half()

        rgb = rgb.reshape(-1, 3, 1) + 0.5
        grd = field_samples.reshape(-1, 1, 3)
        grd = grd / ((grd * grd).sum(dim=-1, keepdim=True) + 1).sqrt()
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
                 use_contraction=True,  # [CONTRACTION] 是否使用空间收缩
                 ext_sdf_fixed=False,  # [EXT-SDF] 外部点SDF是否使用固定值（基于距离）
                 current_sh_deg=2,
                 max_sh_deg=2,
                 ablate_circumsphere=False,
                 ablate_gradient=False,
                 encoding_precision="float16",
                 float32_matmul_precision="high",
                 triangulation_backend="isolated_native",
                 triangulation_timeout_seconds=120.0,
                 initialize_triangulation=True,
                 **kwargs):
        super().__init__()
        self.device = vertices.device
        self.density_offset = density_offset  # [DENSITY-RELATED] 存储密度偏移 (保留用于兼容性)
        self.use_contraction = use_contraction  # [CONTRACTION] 存储是否使用空间收缩
        self.ext_sdf_fixed = ext_sdf_fixed  # [EXT-SDF] 外部点SDF使用固定值
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.dir_offset = torch.tensor([
            [0, 0],
            [math.pi, 0],
            [math.pi/3, math.pi/3],
        ], device=self.device)
        sh_dim = ((1+max_sh_deg)**2-1)*3

        module = iNGPDW(sh_dim, encoding_precision=encoding_precision,
                        float32_matmul_precision=float32_matmul_precision, **kwargs)
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
        self._cached_internal_edges = None  # [EDGE-CACHE] 内部边缓存

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
        self.triangulation_context = {"event": "initialization"}
        self.configure_triangulation(triangulation_backend, triangulation_timeout_seconds,
                                     kwargs.get("output_path"))
        if initialize_triangulation:
            self.update_triangulation()

    def configure_triangulation(self, backend, timeout_seconds, output_path=None):
        old = getattr(self, "_triangulation", None)
        if old is not None:
            old.close()
        if output_path is None:
            import tempfile
            failure_dir = Path(tempfile.mkdtemp(prefix="sdfraster-triangulation-"))
        else:
            failure_dir = Path(output_path) / "triangulation_diagnostics"
        self.triangulation_backend = backend
        self.triangulation_timeout_seconds = float(timeout_seconds)
        self._triangulation = TriangulationBackend(failure_dir, backend, timeout_seconds)

    def close_triangulation(self):
        self._triangulation.close()

    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_internal_edges(self):
        """
        获取内部边（两端都是内部点的边），使用缓存避免重复计算

        Returns:
            edges: (E, 2) 内部边的顶点索引对
        """
        if self._cached_internal_edges is not None:
            return self._cached_internal_edges

        # 提取所有边
        from utils.model_util import extract_edges_from_tets
        edges = extract_edges_from_tets(self.indices)  # (E, 2)

        num_internal = self.num_int_verts
        num_verts = self.vertices.shape[0]

        # 安全检查：确保索引不越界
        valid_edges = (edges[:, 0] < num_verts) & (edges[:, 1] < num_verts)
        if not valid_edges.all():
            edges = edges[valid_edges]

        # 排除包含外部点的边（只保留两端都是内部点的边）
        internal_mask = (edges[:, 0] < num_internal) & (edges[:, 1] < num_internal)
        edges = edges[internal_mask]

        self._cached_internal_edges = edges
        return edges

    def get_circumcenters(self):
        circumcenter =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return circumcenter

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None, compute_sdf=True):
        """
        计算四面体批次的特征

        Args:
            compute_sdf: 是否计算 SDF。渲染时设为 False（使用 query_vert_sdf），其他用途设为 True
        """
        tets = vertices[indices[start:end]]
        if circumcenters is None:
            circumcenter, radius = calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
            radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        if self.ablate_circumsphere:
            circumcenter = tets.mean(dim=1)
        if self.training:
            circumcenter += self.alpha*torch.randn_like(circumcenter) * radius.reshape(-1, 1)
        if self.backbone.k_samples > 1:
            eps = self.sphere_pattern.unsqueeze(0)
            if self.training:
                rand_mat = torch.randn(3, 3, device=circumcenter.device)
                q, _ = torch.linalg.qr(rand_mat)
                if torch.det(q) < 0:
                    q[:, 0] = -q[:, 0]
                eps = torch.matmul(q, eps.transpose(-1, -2)).transpose(-1, -2)

            sampled_cc = circumcenter.reshape(-1, 1, 3) + eps * radius.reshape(-1, 1, 1) * self.backbone.trunc_sigma
            sampled_radius = radius.reshape(-1, 1, 1).expand(-1, self.backbone.k_samples, 1)
            normalized = (sampled_cc.detach() - self.center) / self.scene_scaling
            # [CONTRACTION] 根据 use_contraction 决定是否使用空间收缩
            if self.use_contraction:
                cv, cr = contract_mean_std(normalized.reshape(-1, 3), sampled_radius.reshape(-1) / self.scene_scaling)
                x = (cv/2 + 1)/2
            else:
                # 不使用收缩：直接线性归一化到 [0, 1]
                x = (normalized.reshape(-1, 3) + 1) / 2
                cr = sampled_radius.reshape(-1) / self.scene_scaling
        else:
            normalized = (circumcenter.detach() - self.center) / self.scene_scaling
            # [CONTRACTION] 根据 use_contraction 决定是否使用空间收缩
            if self.use_contraction:
                cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
                x = (cv/2 + 1)/2
            else:
                # 不使用收缩：直接线性归一化到 [0, 1]
                x = (normalized + 1) / 2
                cr = radius / self.scene_scaling

        output = checkpoint(self.backbone, x, cr, compute_sdf, use_reentrant=True)
        # output = (sdf, rgb, grd, sh)，compute_sdf=False 时 sdf 是 dummy 值
        sdf_output = output[0]
        return circumcenter, sdf_output, *output[1:]

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
            # [SDF-MODIFICATION] compute_sdf=False：渲染使用 query_vert_sdf，跳过 sdf_net 计算
            circumcenters, sdf_dummy, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters, compute_sdf=False)
            # === [DENSITY-RELATED] 原密度代码 (已注释) ===
            # circumcenters, density, rgb, grd, sh = self.compute_batch_features(...)
            # === [DENSITY-RELATED] 原密度代码结束 ===
            if self.ablate_gradient:
                grd = torch.zeros_like(grd)
            centroids = vertices[indices[start:end]].mean(dim=1)
            shs[start:end] = sh.reshape(-1, sh_dim, 3)
            # [SDF-MODIFICATION] sdf_dummy 是占位值（渲染不使用 cell_values[0]）
            dvrgbs = activate_output(camera.camera_center.to(self.device),
                                     sdf_dummy, rgb, grd,
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
        init_num = kwargs.pop('init_num', 0)  # 初始点云最小数量
        torch.manual_seed(kwargs.get("seed", 2))

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
                      max_sh_deg=max_sh_deg, **kwargs)
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        runtime_path = path / "runtime_state.json"
        if runtime_path.exists():
            import json
            if json.loads(runtime_path.read_text()).get("frozen"):
                raise ValueError("Frozen-model checkpoint requires its matching model loader")
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
        model = Model(vertices.to(device), ext_vertices, ckpt['center'], ckpt['scene_scaling'],
                      initialize_triangulation=False, **config.as_dict())
        model.load_state_dict(ckpt)
        base_min_t = getattr(config, "base_min_t", getattr(config, "min_t", 0.4))
        model.min_t = model.scene_scaling * base_min_t
        model.indices = torch.as_tensor(indices, device=device)
        model.empty_indices = torch.as_tensor(empty_indices, device=device)
        model.tet_normal_mode = getattr(config, "tet_normal_mode", "raw")
        from utils.checkpoint_util import restore_runtime_state
        restore_runtime_state(model, path / "runtime_state.json")
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

    # [SDF-MODIFICATION] 直接查询顶点SDF值
    def query_vert_sdf(self, points=None):
        """
        直接查询点的SDF值

        Args:
            points: (N, 3) 世界坐标，如果为 None 则查询所有顶点
        Returns:
            sdf: (N,) SDF 值
        """
        # [EXT-SDF] 如果 points=None 且开启 ext_sdf_fixed，只查询内部点
        if points is None and self.ext_sdf_fixed:
            # 只查询内部顶点
            internal_sdf = self._query_sdf_impl(self.contracted_vertices)
            # 外部点使用基于距离的固定值
            ext_normalized = (self.ext_vertices - self.center) / self.scene_scaling
            ext_sdf = torch.norm(ext_normalized, dim=-1)  # 距离中心的归一化距离
            return torch.cat([internal_sdf, ext_sdf])

        if points is None:
            points = self.vertices

        return self._query_sdf_impl(points)

    def _query_sdf_impl(self, points):
        """内部方法：实际执行 SDF 查询"""
        sdf_list = []
        for start in range(0, points.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, points.shape[0])
            pts = points[start:end]

            # 1. 坐标归一化
            normalized = (pts - self.center) / self.scene_scaling

            # 2. [CONTRACTION] 根据 use_contraction 决定是否使用空间收缩
            if self.use_contraction:
                cv, _ = contract_mean_std(normalized, torch.zeros(pts.shape[0], device=pts.device))
                x = (cv / 2 + 1) / 2  # [0, 1] 范围
            else:
                # 不使用收缩：直接线性归一化到 [0, 1]
                x = (normalized + 1) / 2

            # 3. Hash Grid 编码 + C2F
            h = self.backbone.encoding(x)
            if self.backbone.coarse2fine_enabled and self.backbone.active_levels < self.backbone.L:
                h = h * self.backbone._get_c2f_mask(h)
            h = h.float()

            # 4. SDF 网络前向传播
            if self.backbone.use_geometric_init:
                # [GEOMETRIC-INIT] xyz + hash 特征输入
                # xyz 转换到 [-1, 1] 范围（与旧版本一致）
                xyz_centered = x * 2 - 1
                sdf_input = torch.cat([xyz_centered, h], dim=-1)
                sdf_normalized = self.backbone.sdf_net(sdf_input)
                # 将归一化空间的 SDF 转换为世界空间
                sdf = sdf_normalized * self.scene_scaling
            else:
                # 仅 hash 特征输入
                sdf = self.backbone.sdf_net(h)
            sdf_list.append(sdf.reshape(-1))

        return torch.cat(sdf_list)

    def _query_sdf_normalized(self, points):
        """
        在归一化空间查询 SDF（用于 Eikonal Loss）

        当使用几何初始化时，返回未缩放的 SDF（归一化空间），
        使得梯度范数在归一化空间中约为 1。

        Args:
            points: (N, 3) 世界坐标
        Returns:
            sdf_normalized: (N,) 归一化空间的 SDF 值
        """
        sdf_list = []
        for start in range(0, points.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, points.shape[0])
            pts = points[start:end]

            # 1. 坐标归一化
            normalized = (pts - self.center) / self.scene_scaling

            # 2. [CONTRACTION] 根据 use_contraction 决定是否使用空间收缩
            if self.use_contraction:
                cv, _ = contract_mean_std(normalized, torch.zeros(pts.shape[0], device=pts.device))
                x = (cv / 2 + 1) / 2  # [0, 1] 范围
            else:
                # 不使用收缩：直接线性归一化到 [0, 1]
                x = (normalized + 1) / 2

            # 3. Hash Grid 编码 + C2F
            h = self.backbone.encoding(x)
            if self.backbone.coarse2fine_enabled and self.backbone.active_levels < self.backbone.L:
                h = h * self.backbone._get_c2f_mask(h)
            h = h.float()

            # 4. SDF 网络前向传播（不缩放）
            if self.backbone.use_geometric_init:
                xyz_centered = x * 2 - 1
                sdf_input = torch.cat([xyz_centered, h], dim=-1)
                sdf = self.backbone.sdf_net(sdf_input)
            else:
                sdf = self.backbone.sdf_net(h)
            sdf_list.append(sdf.reshape(-1))

        return torch.cat(sdf_list)

    def compute_sdf_gradients(
        self, points=None, mode="numerical", taps=4, eps=None,
        compute_hessian=True, hessian_mode="diagonal",
    ):
        """Compute pointwise SDF gradients and optional second derivatives.

        Analytical diagonal returns [f_xx, f_yy, f_zz].
        legacy_row_sum reproduces Neuralangelo's analytical H @ ones
        for controlled ablations; it is not a Hessian diagonal.
        Numerical formulas and world-space epsilon retain their old behavior:
        six taps returns the diagonal, four taps returns trace / 3 repeated.
        compute_hessian=False skips second derivatives for Eikonal-only
        callers. Analytical evaluation continues to return detached gradients
        and no Hessian, as before.
        """
        if points is None:
            points = self.vertices

        if hessian_mode not in ("diagonal", "legacy_row_sum"):
            raise ValueError(f"unknown hessian_mode: {hessian_mode}")
        if mode == "analytical":
            return self._compute_gradients_analytical(
                points, compute_hessian=compute_hessian, hessian_mode=hessian_mode
            )

        # Preserve the existing world-space finite-difference scale.
        if eps is None:
            finest_resolution = self.backbone.base_resolution * (
                self.backbone.per_level_scale ** (self.backbone.L - 1)
            )
            eps = float(self.scene_scaling) / finest_resolution

        if mode == "numerical":
            if taps == 6:
                return self._compute_gradients_numerical_6(
                    points, eps, compute_hessian=compute_hessian
                )
            if taps == 4:
                return self._compute_gradients_numerical_4(
                    points, eps, compute_hessian=compute_hessian
                )
            raise ValueError(f"taps must be 4 or 6, got {taps}")
        raise ValueError(f"mode must be 'analytical' or 'numerical', got {mode}")

    def _compute_gradients_analytical(
        self, points, compute_hessian=True, hessian_mode="diagonal",
    ):
        """Differentiate each first derivative along its matching coordinate."""
        if hessian_mode not in ("diagonal", "legacy_row_sum"):
            raise ValueError(f"unknown hessian_mode: {hessian_mode}")
        with torch.enable_grad():
            # Do not change the caller's requires_grad flag or vertex graph.
            query_points = points.detach().requires_grad_(True)
            sdf = self.query_vert_sdf(query_points)
            if sdf.requires_grad:
                gradients = torch.autograd.grad(
                    sdf.sum(), query_points, create_graph=self.training,
                    allow_unused=True,
                )[0]
            else:
                gradients = None
            if gradients is None:
                gradients = query_points * 0.0 + sdf.unsqueeze(-1) * 0.0

            hessian_diag = None
            if self.training and compute_hessian:
                # Zero curvature for affine/constant components must also be
                # safe to backpropagate through when parameters are present.
                zero = gradients * 0.0 + sdf.unsqueeze(-1) * 0.0

                def second_derivative(component):
                    if not component.requires_grad:
                        return None
                    return torch.autograd.grad(
                        component.sum(), query_points, create_graph=True,
                        retain_graph=True, allow_unused=True,
                    )[0]

                if hessian_mode == "legacy_row_sum":
                    row_sum = second_derivative(gradients.sum(dim=-1))
                    hessian_diag = zero if row_sum is None else row_sum + zero
                else:
                    diagonal = []
                    for axis in range(query_points.shape[-1]):
                        row = second_derivative(gradients[..., axis])
                        entry = zero[..., axis] if row is None else row[..., axis] + zero[..., axis]
                        diagonal.append(entry)
                    hessian_diag = torch.stack(diagonal, dim=-1)

            if not self.training:
                return sdf.detach(), gradients.detach(), None
            return sdf, gradients, hessian_diag

    def _compute_gradients_numerical_6(self, points, eps, compute_hessian=True):
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

        hessian_diag = None
        if compute_hessian:
            # Six taps estimates each diagonal element independently.
            hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)
            hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)
            hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)
            hessian_diag = torch.stack([hessian_xx, hessian_yy, hessian_zz], dim=-1)

        return sdf, gradients, hessian_diag

    def _compute_gradients_numerical_4(self, points, eps, compute_hessian=True):
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

        hessian_diag = None
        if compute_hessian:
            # Four taps only estimates the trace; these are not individual
            # diagonal entries. Preserve Neuralangelo's trace / 3 convention.
            hessian_trace = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / (eps_adj ** 2)
            hessian_diag = torch.stack([hessian_trace, hessian_trace, hessian_trace], dim=-1) / 3.0

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
    def active_internal_vertices(self):
        """获取活跃的内部顶点（排除 ext_vertices，用于 Eikonal loss）"""
        active_indices = self.indices.unique()
        num_internal = self.contracted_vertices.shape[0]
        # 只保留索引 < num_internal 的顶点（即 contracted_vertices）
        internal_mask = active_indices < num_internal
        internal_indices = active_indices[internal_mask]
        return self.contracted_vertices[internal_indices]

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
        if not 0 <= sdf_eps < 0.5 or not 0 <= alpha_threshold <= 1:
            raise ValueError("Invalid triangulation culling thresholds")
        torch.cuda.empty_cache()
        verts = self.vertices
        points = verts.detach().cpu().double().numpy()
        indices_np, metadata = self._triangulation.compute(
            points, high_precision=high_precision, context=self.triangulation_context)
        # Orientation and geometry have already been checked with float64 source coordinates.
        indices = torch.as_tensor(indices_np, device=verts.device)
        mask = None
        if sdf_eps > 0 or alpha_threshold > 0:
            vert_sdf = self.query_vert_sdf()
            tet_vert_sdf = vert_sdf[indices.long()]
            if sdf_eps > 0:
                inv_s = self.backbone.inv_s
                k = math.log((1 - sdf_eps) / sdf_eps)
                sdf_threshold = k / inv_s
                sdf_mask = tet_vert_sdf.abs().min(dim=1)[0] < sdf_threshold
            else:
                sdf_mask = torch.zeros(len(indices), dtype=torch.bool, device=indices.device)
            if alpha_threshold > 0:
                tet_alpha = self.calc_tet_alpha(indices=indices, vert_sdf=vert_sdf)
                alpha_mask = tet_alpha > alpha_threshold
            else:
                alpha_mask = torch.ones(len(indices), dtype=torch.bool, device=indices.device)
            mask = sdf_mask | alpha_mask
            empty_indices = indices[~mask]
            kept_indices = indices[mask]
        else:
            empty_indices = torch.empty((0, 4), dtype=indices.dtype, device=indices.device)
            kept_indices = indices
        # Do not publish a candidate topology until validation and field culling both succeed.
        self.indices = kept_indices
        self.empty_indices = empty_indices
        if mask is not None:
            self.mask = mask
        self._cached_internal_edges = None
        self.triangulation_last_metadata = metadata
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

    def _query_sdf_normalized_space(self, points_normalized):
        """
        直接在归一化空间查询 SDF，不乘 scaling（用于调试坐标变换问题）

        Args:
            points_normalized: (N, 3) 归一化坐标 [-1, 1]
        Returns:
            sdf_normalized: (N,) 归一化空间的 SDF（不乘 scaling）
        """
        sdf_list = []
        for start in range(0, points_normalized.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, points_normalized.shape[0])
            pts_norm = points_normalized[start:end]

            # 归一化 → hash [0, 1]
            x = (pts_norm + 1) / 2

            # Hash Grid 编码 + C2F
            h = self.backbone.encoding(x)
            if self.backbone.coarse2fine_enabled and self.backbone.active_levels < self.backbone.L:
                h = h * self.backbone._get_c2f_mask(h)
            h = h.float()

            # SDF 网络输入
            xyz_input = x * 2 - 1  # = pts_norm
            sdf_input = torch.cat([xyz_input, h], dim=-1)

            # SDF 网络输出（不乘 scaling！）
            sdf_normalized = self.backbone.sdf_net(sdf_input)
            sdf_list.append(sdf_normalized.reshape(-1))

        return torch.cat(sdf_list)

    def _compute_eikonal_normalized_space(self, points_normalized, eps):
        """
        在归一化空间用数值差分计算 Eikonal loss（用于调试坐标变换问题）

        Args:
            points_normalized: (N, 3) 归一化坐标 [-1, 1]
            eps: 归一化空间的步长 (通常 = 1/resolution)
        Returns:
            sdf: (N,) SDF 值（归一化空间）
            gradients: (N, 3) 梯度
            eikonal_loss: scalar
        """
        # 中心点 SDF
        sdf = self._query_sdf_normalized_space(points_normalized)

        # 6 个偏移方向
        eps_x = torch.tensor([eps, 0., 0.], device=self.device)
        eps_y = torch.tensor([0., eps, 0.], device=self.device)
        eps_z = torch.tensor([0., 0., eps], device=self.device)

        # 计算 6 个偏移点的 SDF
        sdf_x_pos = self._query_sdf_normalized_space(points_normalized + eps_x)
        sdf_x_neg = self._query_sdf_normalized_space(points_normalized - eps_x)
        sdf_y_pos = self._query_sdf_normalized_space(points_normalized + eps_y)
        sdf_y_neg = self._query_sdf_normalized_space(points_normalized - eps_y)
        sdf_z_pos = self._query_sdf_normalized_space(points_normalized + eps_z)
        sdf_z_neg = self._query_sdf_normalized_space(points_normalized - eps_z)

        # 中心差分
        gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
        gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
        gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
        gradients = torch.stack([gradient_x, gradient_y, gradient_z], dim=-1)

        grad_norms = gradients.norm(dim=-1)
        eikonal_loss = ((grad_norms - 1) ** 2).mean()

        return sdf, gradients, eikonal_loss

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
                 lr_horizon: int = 0,
                 lr_mode: str = "legacy_spiking",
                 lr_peak_start: int = -1,

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
        # [SDF-MODIFICATION] s_var 学习率倍率（从 kwargs 读取，默认 1）
        self.s_var_lr_multiplier = kwargs.get('s_var_lr_multiplier', 1)
        if model.backbone.s_var_learn:
            # 初始 lr 会被 update_learning_rate 立即覆盖，这里只是占位
            net_param_groups.append({"params": [model.backbone.s_var], "lr": network_lr, "name": "s_var"})

        self.net_optim = optim.CustomAdam(net_param_groups, ignore_param_list=[], betas=[0.9, 0.999], eps=1e-15)
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.lr_policy = resolve_optimizer_lr(
            freeze_start, lr_horizon, lr_mode, lr_peak_start, midpoint)
        effective_lr_horizon = self.lr_policy["effective_lr_horizon"]
        effective_peak_start = self.lr_policy["effective_lr_peak_start"]
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
                                                max_steps=effective_lr_horizon)

        self.net_scheduler_args = SpikingLR(
            spike_duration, effective_lr_horizon, base_net_scheduler,
            effective_peak_start, densify_interval, densify_end,
            network_lr, network_lr)
            # network_lr, final_network_lr)

        base_encoder_scheduler = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=effective_lr_horizon)

        self.encoder_scheduler_args = SpikingLR(
            spike_duration, effective_lr_horizon, base_encoder_scheduler,
            effective_peak_start, densify_interval, densify_end,
            # encoding_lr, encoding_lr)
            encoding_lr, encoding_lr)

        self.vertices_lr = self.vert_lr_multi*vertices_lr
        self.final_vertices_lr = self.vert_lr_multi*final_vertices_lr
        base_vertex_scheduler = get_expon_lr_func(lr_init=self.vertices_lr,
                                                lr_final=self.final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=effective_lr_horizon,
                                                lr_delay_steps=vert_lr_delay)

        self.vertex_scheduler_args = base_vertex_scheduler
        self.vertex_scheduler_args = SpikingLR(
            spike_duration, effective_lr_horizon, base_vertex_scheduler,
            effective_peak_start, densify_interval, densify_end,
            # self.vertices_lr, self.final_vertices_lr)
            self.vertices_lr, self.vertices_lr)
        if lr_mode == "exponential":
            self.net_scheduler_args = base_net_scheduler
            self.encoder_scheduler_args = base_encoder_scheduler
            self.vertex_scheduler_args = base_vertex_scheduler
        self.iteration = 0

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.iteration = iteration
        self.model.alpha = self.alpha_sched(iteration)
        for param_group in self.net_optim.param_groups:
            # [SDF-MODIFICATION] s_var 使用独立的学习率倍率
            if param_group.get("name") == "s_var":
                lr = self.net_scheduler_args(iteration) * self.s_var_lr_multiplier
            else:
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
        self.model.triangulation_context = {**self.model.triangulation_context, "event": "densify"}
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        self.model.triangulation_context = {**self.model.triangulation_context, "event": "prune"}
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

    # [PRUNE] 基于贡献度 + 表面距离的顶点剪枝
    def prune(self, peak_contrib=None,
              prune_contrib_threshold=0.01, prune_sdf_far_threshold=0.1, **kwargs):
        """
        剪枝策略：移除同时满足以下条件的顶点：
        1. 贡献度低（对渲染贡献小）
        2. 远离表面（SDF绝对值大）
        """
        indices = self.model.indices.long()
        device = indices.device
        n_verts = self.model.vertices.shape[0]
        n_tets = indices.shape[0]
        num_internal = self.model.num_int_verts

        # 计算顶点到表面的距离
        vert_sdf = self.model.query_vert_sdf()
        far_from_surface = vert_sdf.abs() > prune_sdf_far_threshold

        # 计算顶点贡献度（四面体贡献度聚合到顶点）
        if peak_contrib is not None:
            vert_contrib = torch.zeros((n_verts,), device=device)
            vert_contrib.scatter_reduce_(dim=0, index=indices[..., 0], src=peak_contrib, reduce="amax")
            vert_contrib.scatter_reduce_(dim=0, index=indices[..., 1], src=peak_contrib, reduce="amax")
            vert_contrib.scatter_reduce_(dim=0, index=indices[..., 2], src=peak_contrib, reduce="amax")
            vert_contrib.scatter_reduce_(dim=0, index=indices[..., 3], src=peak_contrib, reduce="amax")
            low_contrib = vert_contrib < prune_contrib_threshold
        else:
            low_contrib = torch.ones((n_verts,), dtype=torch.bool, device=device)

        # 剪枝条件：贡献度低 且 远离表面，只剪枝内部顶点
        prune_mask = low_contrib & far_from_surface
        prune_mask[num_internal:] = False
        keep_mask = ~prune_mask

        n_pruned = prune_mask.sum().item()
        print(f"Pruned {n_pruned} points. "
              f"low_contrib: {low_contrib[:num_internal].sum().item()}, "
              f"far_from_surface: {far_from_surface[:num_internal].sum().item()}")

        if n_pruned > 0:
            self.remove_points(keep_mask)
