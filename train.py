import cv2
import os
# VERSION = 9
# if VERSION is not None:
#     os.environ["CC"] = f"/usr/bin/gcc-{VERSION}"
#     os.environ["CXX"] = f"/usr/bin/g++-{VERSION}"
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))
print(str(Path(os.path.abspath('')).parent))
import math
import torch
from data import loader
import random
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import open3d as o3d

# [SEED] 设置随机种子，保证可复现性
SEED = 52
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

from utils import cam_util
from utils.train_util import *
from utils.model_util import eikonal_loss, curvature_loss, sample_eikonal_points, edge_sdf_loss, gradient_tv_loss, gradient_tv_loss_from_sampled, gradient_tv_loss_surface_edges
# from models.vertex_color import Model, TetOptimizer
from models.ingp_color import Model, TetOptimizer
from models.frozen import freeze_model
# from models.ingp_density import Model, TetOptimizer
# from models.frozen_vertices import freeze_model
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import json
import imageio
from utils import test_util
import termplotlib as tpl
import matplotlib.pyplot as plt

# [DEBUG-MEMORY] 显存调试开关
DEBUG_MEMORY = False  # 设为 True 开启显存监控
if DEBUG_MEMORY:
    from utils.memory_monitor import MemoryMonitor, print_gpu_memory, print_large_tensors

def viz_tensordepth(x, alpha_mass=None, mode='log'):
    """深度图可视化函数，参考 svraster 实现"""
    # [DEBUG] 打印 depth tensor 的统计信息
    print(f"[DEBUG viz_tensordepth] depth: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
    if alpha_mass is not None:
        print(f"[DEBUG viz_tensordepth] alpha: min={alpha_mass.min():.6f}, max={alpha_mass.max():.6f}")
        m = (alpha_mass > 0.01) & (x > 0)
    else:
        m = (x > 0)
    print(f"[DEBUG viz_tensordepth] valid pixels (m.sum): {m.sum()}")

    x = x.cpu().numpy()
    m = m.cpu().numpy()
    if m.sum() == 0:
        h, w = x.shape[-2:]
        return np.zeros((h, w, 3), dtype=np.uint8)
    dmin, dmax = np.quantile(x[m], q=[0.03, 0.97])
    x = np.log(np.clip(1 + x - dmin, 1, 1e9))
    x = x / np.log(1 + dmax - dmin)
    x = np.clip(x, 0, 1) * 255
    viz = cv2.applyColorMap(x.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
    viz[~m] = 0
    return viz
from utils.lib_bilagrid import BilateralGrid, total_variation_loss, slice
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ChainedScheduler
import gc
from utils.densification import collect_render_stats, apply_densification, determine_cull_mask
from utils.mesh_regularization import MeshRegularizer
from utils.tetmesh import marching_tetrahedra
from utils.mesh import Meshes, save_mesh_as_ply
from utils.geometry_utils import depth_to_normal, compute_normal_consistency_loss, fix_normal_map
import mediapy
from icecream import ic

torch.set_num_threads(1)

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)

class SimpleSampler:
    def __init__(self, total_num_samples, batch_size):
        self.total_num_samples = total_num_samples
        self.batch_size = batch_size
        self.curr = total_num_samples
        self.ids = None

    def nextids(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self.curr += batch_size
        if self.curr + batch_size > self.total_num_samples:
            # self.ids = torch.LongTensor(np.random.permutation(self.total_num_samples))
            self.ids = torch.randperm(self.total_num_samples, dtype=torch.long, device=device)
            self.curr = 0
        ids = self.ids[self.curr : self.curr + batch_size]
        return ids

eps = torch.finfo(torch.float).eps
args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.eval = False
args.dataset_path = Path("/optane/nerf_datasets/360/bonsai")
args.output_path = Path("output/test/")
args.iterations = 18000
args.ckpt = ""
args.resolution = 1
args.render_train = False

# Light Settings
args.max_sh_deg = 3
args.sh_interval = 0
args.sh_step = 1

# iNGP Settings
args.use_tcnn = True  # True: 使用 tiny-cuda-nn (更快), False: 使用 PyTorch 实现
args.use_contraction = True  # True: 使用 Mip-NeRF 360 空间收缩, False: 仅线性归一化 (推荐关闭以正确计算 eikonal)
args.ext_sdf_fixed = False   # True: 外部点SDF使用固定值（基于距离），False: 通过哈希网格查询
# [GEOMETRIC-INIT] Neuralangelo 风格的几何初始化
args.use_geometric_init = False  # True: xyz输入+几何初始化+SDF缩放, False: 仅hash特征（默认）
args.sdf_bias = 0.5              # 几何初始化的 SDF 偏移（控制初始球体半径）
args.sdf_inside_out = False      # 是否反转 SDF 内外
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 1e-3
args.final_network_lr = 1e-4
args.hidden_dim = 64
args.scale_multi = 0.35 # chosen such that 96% of the distribution is within the sphere
args.log2_hashmap_size = 22
args.per_level_scale = 2
args.L = 8
args.hashmap_dim = 8
args.base_resolution = 64

# Coarse-to-Fine 策略 (参考 Neuralangelo)
args.coarse2fine_enabled = False  # 是否启用 coarse2fine
args.coarse2fine_init_level = 4   # 初始激活的 hash grid 层数
args.coarse2fine_step = 3000      # 每隔多少 iteration 增加一层

args.density_offset = -4
args.lambda_weight_decay = 1
args.percent_alpha = 0.0 # preconditioning
args.spike_duration = 500

args.g_init=1.0
args.s_init=1e-4
args.d_init=0.1
args.c_init=0.8

# Vertex Settings
args.lr_delay = 0
args.vert_lr_delay = 0
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.delaunay_interval = 10

args.freeze_start = 18000
args.freeze_lr = 1e-3
args.final_freeze_lr = 1e-4

# Distortion Settings
args.lambda_dist = 0.0
args.lambda_sh = 0.0

# [SDF-MODIFICATION] NeuS s_var 学习率设置
args.s_var_lr_multiplier = 10   # s_var 学习率倍率（相对于 network_lr）

# [SDF-MODIFICATION] Eikonal 和 Curvature Loss 设置（参考 Neuralangelo）
args.lambda_eikonal = 0.0 #0.1       # Eikonal loss 权重
args.lambda_curvature = 0.0 #5e-4   # Curvature loss 权重
args.eikonal_start = 0            # 开始使用 eikonal loss 的迭代
args.eikonal_end = 30000          # 停止使用 eikonal loss 的迭代
args.curvature_start = 0         # 开始使用 curvature loss 的迭代
args.curvature_end = 18000        # 停止使用 curvature loss 的迭代
args.curvature_warmup = 2000      # curvature loss warmup 迭代数（从 curvature_start 开始）
args.curvature_decay_enabled = True  # 是否启用 curvature 权重衰减
args.gradient_mode = "analytical"  # "analytical" 或 "numerical"
args.gradient_taps = 4            # 数值梯度采样点数：4 或 6
args.gradient_eps = 1e-3          # 数值梯度步长
args.num_eikonal_samples = 500000  # Eikonal loss 采样点数
args.eikonal_band_threshold = None # SDF band 阈值：仅计算 |SDF| < threshold 的顶点的 loss
                                   # 设为 None 则计算所有内部顶点
args.eikonal_tet_sampling = False  # 是否在 band 内顶点邻域额外采样
                                   # True: band内顶点 + 邻域采样; False: 仅 band内顶点

# [EDGE-SDF-LOSS] Edge SDF Loss 设置（参考 svraster TV loss）
args.lambda_edge_sdf = 0.0        # Edge SDF loss 权重，0 表示关闭
args.edge_sdf_start = 0           # 开始使用 edge sdf loss 的迭代
args.edge_sdf_end = 30000         # 停止使用 edge sdf loss 的迭代
args.edge_sdf_threshold = 0.1     # SDF band 阈值：只计算至少一端 |SDF| < threshold 的边
args.edge_length_threshold = None # 边长阈值：只计算长度 < threshold 的边，None 表示不限制
args.edge_sdf_target = None       # 目标值：None=平滑模式(最小化), 1.0=Eikonal模式(约束接近1)

# [GRADIENT-TV-LOSS] Gradient TV Loss 设置（促进SDF梯度平滑）
args.lambda_gradient_tv = 0.0     # Gradient TV loss 权重，0 表示关闭
args.gradient_tv_start = 0        # 开始使用 gradient TV loss 的迭代
args.gradient_tv_end = 30000      # 停止使用 gradient TV loss 的迭代
args.gradient_tv_surface_only = False   # True=只计算穿过表面的边（SDF异号），False=计算所有边
args.gradient_tv_normalize = True      # 是否用边长归一化梯度差异

# [MESH-NORMAL-LOSS] Mesh 法线一致性 Loss 设置（参考 MILo）
args.lambda_mesh_normal = 0.0   # mesh normal loss 权重
args.lambda_mesh_depth = 0.0     # mesh depth loss 权重
args.lambda_mesh_occupancy = 0.0 # surface occupancy loss 权重
args.lambda_mesh_occupancy_labels = 0.0  # occupancy labels loss 权重
args.mesh_occupancy_labels_reset_interval = 200  # 重新计算 occupancy labels 的间隔
args.mesh_depth_scale = None     # depth loss 的 scale 参数 (None则使用spatial_lr_scale)
args.mesh_depth_ratio = 1.0      # MILo: median深度的混合比例 (0.0=仅期望深度, 1.0=仅中位深度)
args.mesh_normal_start = 5001     # 开始使用 mesh normal loss 的迭代
args.mesh_normal_stop = 18000     # 停止使用 mesh normal loss 的迭代
args.mesh_sdf_threshold = 0.0     # mesh 提取的 SDF 阈值
args.mesh_normal_source = "depth_med"  # 计算 gaussian normal 的来源:
                                   # "depth": 使用期望深度推导法线 (默认)
                                   # "depth_med": 使用中位深度推导法线
                                   # "rendered": 直接使用渲染的法线（跳过 depth_to_normal）

# [DEPTH-NORMAL-LOSS] 深度推导法线与渲染法线一致性 Loss（参考 MILo）
args.lambda_normal_depth = 0.0      # 渲染的normal vs 期望深度推导的normal
args.lambda_normal_depth_med = 0.0  # 渲染的normal vs 中位深度推导的normal
args.normal_depth_start = 5001         # 开始使用 normal-depth loss 的迭代
args.normal_depth_end = 30000       # 停止使用 normal-depth loss 的迭代

# [MASK-LOSS] Mask Loss 和 Transmittance Concentration Loss（参考 SVRaster）
args.lambda_mask = 0.1              # mask loss 权重：约束 T 与 GT mask 对齐
args.mask_from = 0                  # 开始使用 mask loss 的迭代
args.mask_until = 20000             # 停止使用 mask loss 的迭代
args.lambda_T_concen = 0.0          # transmittance concentration loss：鼓励 T 二值化到 0 或 1

# Clone Settings
args.num_samples = 200
args.k_samples = 1
args.trunc_sigma = 0.35
args.min_tet_count = 9
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500
args.budget = 2_000_000
args.max_split_per_iter = 50000         # 每次分裂的最大顶点数
args.within_thresh = 0.5
args.total_thresh = 2.0
args.clone_min_contrib = 0.003
args.split_min_contrib = 0.01

# [SDF-DENSIFY] SDF 导向的分裂策略
# 数量 = 当前顶点数 × ratio，选择表面附近外接圆最大的 TopK，分裂点为重心
args.sdf_densify_enabled = False      # 是否启用 SDF 导向的分裂
args.sdf_densify_threshold = 0.02     # |SDF| < 此值认为接近表面
args.sdf_densify_ratio = 0.2          # SDF 分裂数量 = 当前顶点数 × ratio
args.sdf_densify_start = 4000         # 开始使用 SDF 分裂的迭代数
args.sdf_densify_end = 10000          # 停止使用 SDF 分裂的迭代数

# [PRUNE] 顶点剪枝策略：移除 (贡献度低 且 远离表面) 的顶点
args.prune_enabled = False               # 是否启用顶点剪枝
args.prune_start = 4000                 # 开始剪枝的迭代数
args.prune_end = 15000                  # 停止剪枝的迭代数
args.prune_contrib_threshold = 0.01     # 贡献度阈值，低于此值认为贡献度低
args.prune_sdf_far_threshold = 0.1      # SDF距离阈值，大于此值认为远离表面

args.lambda_ssim = 0.2
args.min_t = 0.4
args.sample_cam = 8
args.data_device = 'cpu'
# [SDF-MODIFICATION] sdf_eps: CDF 贡献阈值 ε
# 保留 CDF 贡献 > ε 的四面体，例如 0.01 表示保留贡献 > 1% 的
args.sdf_eps = 0.01
args.alpha_threshold = 0.1
# [CULL] 四面体剔除：contrib_threshold 控制剔除阈值
args.contrib_threshold = 0.0
args.threshold_start = 4500             # 开始剔除的迭代数（iteration > threshold_start 时启用）
args.voxel_size = 0.01
args.sdf_sphere_radius_scale = 1.2  # SDF球形初始化：点云半径 × scale / scene_scaling
args.init_num = 0  # 初始点云最小数量，不足则通过扰动扩充

args.ablate_gradient = False
args.ablate_circumsphere = True
args.ablate_downweighing = False

# 渲染图保存设置
args.save_render_interval = 500  # 每k次迭代保存渲染图，0表示不保存
args.save_mesh_interval = 1000   # 每k次迭代用MT提取mesh并保存，0表示不保存

# [MESH-EVAL] Mesh 提取与评估设置（复用 MILo 评估代码）
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 路径1: MT提取 → cull → 相机位置对齐 → 评估                          │
# │ 路径2: TSDF提取 → cull → 相机位置对齐 → 评估                        │
# │                                                                     │
# │ DTU: 使用 transform.txt + scale_mat 对齐                           │
# │ TNT: 使用 transform.txt + 轨迹对齐 (RANSAC + ICP)                   │
# └─────────────────────────────────────────────────────────────────────┘
args.eval_dtu = True            # 是否进行 DTU 评估（仅对 DTU 数据集有效）
args.eval_tnt = True            # 是否进行 TNT 评估（仅对 TNT 数据集有效）
args.eval_mt = True             # 是否评估 MT 提取的 mesh
args.extract_tsdf = True        # 是否使用 TSDF Fusion 提取并评估 mesh
args.cull_dilation_radius = 6   # cull_mesh 时 mask 膨胀半径（MILo 默认 6）
args.tnt_min_visible_views = 20 # TNT cull_mesh 时最少可见视图数

# [TSDF-FUSION] TSDF Fusion 参数（参考 MILo DTU 配置）
args.tsdf_depth_ratio = 0.0     # 深度混合比例: 0.0 = 仅中位深度, 1.0 = 仅期望深度
args.tsdf_voxel_size = 0.002    # 体素大小
args.tsdf_depth_trunc = 8.0     # 深度截断
args.tsdf_sdf_trunc = 0.01      # SDF 截断
args.tsdf_num_cluster = 50      # 后处理保留的最大 cluster 数


args = Args.from_namespace(args.get_parser().parse_args())

# [EIKONAL-FIX] 几何初始化时自动禁用 contraction
# contraction 是非线性变换，会破坏 eikonal loss 的梯度缩放
if args.use_geometric_init and args.use_contraction:
    print("[EIKONAL-FIX] use_geometric_init=True, 自动禁用 contraction")
    args.use_contraction = False

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=args.eval, resolution=args.resolution)

# [MILo] 计算 spatial_lr_scale（MILo 风格：相机阵列半径 × 1.1）
cam_centers = torch.stack([c.camera_center.reshape(3) for c in train_cameras], dim=0)
avg_cam_center = cam_centers.mean(dim=0, keepdim=True)
cam_dists = torch.norm(cam_centers - avg_cam_center, dim=1)
spatial_lr_scale = float(cam_dists.max() * 1.1)
print(f"[MILo] spatial_lr_scale: {spatial_lr_scale:.4f}")

np.savetxt(str(args.output_path / "transform.txt"), scene_info.transform)

# 保存相机轨迹（用于 TNT 评估）
# 保存原始坐标系下的 c2w 矩阵（PCA 变换前）
pca_inv = np.linalg.inv(scene_info.transform)
poses = []
for cam in train_cameras:
    # 获取 PCA 变换后的 c2w
    w2c = cam.world_view_transform.T.cpu().numpy()
    c2w = np.linalg.inv(w2c)
    # 逆 PCA 变换回原始坐标系
    c2w_orig = pca_inv @ c2w
    poses.append(c2w_orig)
poses = np.array(poses)
np.save(str(args.output_path / "poses.npy"), poses)
print(f"[MILo] Saved camera poses: {poses.shape}")

args.num_samples = min(len(train_cameras), args.num_samples)

with (args.output_path / "config.json").open("w") as f:
    json.dump(args.as_dict(), f, cls=CustomEncoder)

device = torch.device('cuda')
if len(args.ckpt) > 0: 
    model = Model.load_ckpt(Path(args.ckpt), device)
else:
    model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                                current_sh_deg = args.max_sh_deg if args.sh_interval <= 0 else 0,
                                **args.as_dict())
min_t = args.min_t

tet_optim = TetOptimizer(model, **args.as_dict())

# [MESH-LOSS] 创建 Mesh 正则化器
mesh_regularizer = MeshRegularizer(
    start_iter=args.mesh_normal_start,
    stop_iter=args.mesh_normal_stop,
    normal_weight=args.lambda_mesh_normal,
    depth_weight=args.lambda_mesh_depth,
    occupancy_weight=args.lambda_mesh_occupancy,
    occupancy_labels_weight=args.lambda_mesh_occupancy_labels,
    occupancy_labels_reset_interval=args.mesh_occupancy_labels_reset_interval,
    depth_scale=args.mesh_depth_scale,
    depth_ratio=args.mesh_depth_ratio,  # MILo: 期望/中位深度混合比例
    sdf_threshold=args.mesh_sdf_threshold,
    use_opengl=True,
)

# [DEBUG-MEMORY] 初始化显存监控
if DEBUG_MEMORY:
    memory_monitor = MemoryMonitor(
        log_path=args.output_path / "memory.log",
        print_interval=500  # 每500次迭代打印一次
    )
    print_gpu_memory("[Init] After model creation:")

if args.eval:
    sample_camera = test_cameras[args.sample_cam]
    # sample_camera = train_cameras[args.sample_cam]
else:
    sample_camera = train_cameras[args.sample_cam]

camera_inds = {}
camera_inds_back = {}
for i, camera in enumerate(train_cameras):
    camera_inds[camera.uid] = i
    camera_inds_back[i] = camera.uid

images = []
psnrs = [[]]
inds = []

num_densify_iter = args.densify_end - args.densify_start
N = num_densify_iter // args.densify_interval + 1
S = model.vertices.shape[0]

dschedule = list(range(args.densify_start, args.densify_end, args.densify_interval))

print("Encoding LR")
# xs = list(range(args.iterations))
# ys = [tet_optim.encoder_scheduler_args(x) for x in xs]
# fig = tpl.figure()
# fig.plot(xs, ys, width=150, height=20)
# fig.show()

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples)

# video_writer = cv2.VideoWriter(str(args.output_path / "training.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30,
#                                pad_hw2even(sample_camera.image_width, sample_camera.image_height))

# 创建渲染图保存目录
render_save_dir = args.output_path / "renders"
if args.save_render_interval > 0:
    render_save_dir.mkdir(exist_ok=True, parents=True)

# 创建mesh保存目录
mesh_save_dir = args.output_path / "meshes"
if args.save_mesh_interval > 0:
    mesh_save_dir.mkdir(exist_ok=True, parents=True)

progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    do_delaunay = iteration % args.delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = iteration in dschedule
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0

    # Coarse-to-Fine: 更新激活的 hash grid 层数
    if args.coarse2fine_enabled:
        model.backbone.set_active_levels(iteration)

    if do_delaunay or do_freeze:
        st = time.time()
        # [CULL] 四面体剔除：iteration > threshold_start 时启用
        tet_optim.update_triangulation(
            sdf_eps=args.sdf_eps if iteration > args.threshold_start else 0,
            alpha_threshold=args.alpha_threshold if iteration > args.threshold_start else 0, high_precision=do_freeze)
        if do_freeze:
            del tet_optim
            # model.eval()
            # mask = determine_cull_mask(train_cameras, model, args, device)
            n_tets = model.indices.shape[0]
            mask = torch.ones((n_tets), device=device, dtype=bool)
            # model.train()
            print(f"Kept {mask.sum()} tets")
            model, tet_optim = freeze_model(model, mask, args)
            # model, tet_optim = freeze_model(model, **args.as_dict())
            gc.collect()
            torch.cuda.empty_cache()

    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    camera = train_cameras[ind]
    target = camera.original_image.cuda()
    gt_mask = camera.gt_alpha_mask.cuda()

    # [DEBUG-MEMORY] 渲染前
    if DEBUG_MEMORY:
        memory_monitor.log(iteration, "before_render", model)

    st = time.time()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)

    # [DEBUG-MEMORY] 捕获 OOM 异常并打印详细信息
    try:
        render_pkg = render(camera, model, scene_scaling=model.scene_scaling, ray_jitter=ray_jitter, **args.as_dict())
    except RuntimeError as e:
        if "out of memory" in str(e) and DEBUG_MEMORY:
            print(f"\n{'='*60}")
            print(f"[OOM] Out of memory at iteration {iteration}")
            print(f"{'='*60}")
            print_gpu_memory("Current state:")
            print_large_tensors(min_size_mb=5)
            print(f"\nModel stats: #V={model.vertices.shape[0]}, #T={model.indices.shape[0]}")
            memory_monitor.summary()
        raise  # 重新抛出异常

    image = render_pkg['render']

    # [DEBUG-MEMORY] 渲染后
    if DEBUG_MEMORY:
        memory_monitor.log_delta(iteration, "after_render", model)

    l1_loss = ((target - image).abs() * gt_mask).mean()
    l2_loss = ((target - image)**2 * gt_mask).mean()
    reg = tet_optim.regularizer(render_pkg, **args.as_dict())
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    dl_loss = render_pkg['distortion_loss']

    # [SDF-MODIFICATION] Eikonal 和 Curvature Loss
    eik_loss = torch.tensor(0.0, device=device)
    curv_loss = torch.tensor(0.0, device=device)
    eik_num_samples = 0  # 追踪 band 内的采样点数
    curv_weight_decay = 1.0  # curvature 权重衰减因子
    eik_vertex_indices = None  # 用于 gradient TV loss
    eik_gradients = None  # 用于 gradient TV loss

    # 检查迭代范围
    eikonal_active = args.lambda_eikonal > 0 and args.eikonal_start <= iteration < args.eikonal_end
    curvature_active = args.lambda_curvature > 0 and args.curvature_start <= iteration < args.curvature_end
    if eikonal_active or curvature_active:
        # [NEW-SAMPLING] 使用新的采样策略：band内顶点 + 邻域采样（可选）
        sampled_points, eik_vertex_indices, num_band, num_tet_samples = sample_eikonal_points(
            model,
            num_samples=args.num_eikonal_samples,
            band=args.eikonal_band_threshold,  # None 表示使用所有内部顶点
            use_tet_sampling=args.eikonal_tet_sampling  # False 则仅使用 band 内顶点
        )
        eik_num_samples = sampled_points.shape[0]

        # 如果采样点为空，跳过 loss 计算
        if sampled_points.shape[0] > 0:
            # 在采样点上计算 SDF 梯度
            _, gradients, hessian_diag = model.compute_sdf_gradients(
                points=sampled_points,
                mode=args.gradient_mode,
                taps=args.gradient_taps,
                eps=args.gradient_eps
            )
            eik_gradients = gradients  # 保存梯度用于 gradient TV loss

            if eikonal_active:
                eik_loss = eikonal_loss(gradients)
            if curvature_active and hessian_diag is not None:
                curv_loss = curvature_loss(hessian_diag)
                # [Neuralangelo] curvature 权重调度
                if hasattr(model.backbone, 'get_curvature_weight_decay'):
                    curv_weight_decay = model.backbone.get_curvature_weight_decay(
                        iteration=iteration,
                        warmup_end=args.curvature_start + args.curvature_warmup,
                        decay_end=args.curvature_end,
                        decay_enabled=args.curvature_decay_enabled
                    )

    # [EDGE-SDF-LOSS] Edge SDF Loss（参考 svraster TV loss）
    # [GRADIENT-TV-LOSS] Gradient TV Loss（促进SDF梯度平滑）
    edge_loss = torch.tensor(0.0, device=device)
    edge_num_edges = 0
    grad_tv_loss = torch.tensor(0.0, device=device)
    grad_tv_num_edges = 0
    edge_sdf_active = (args.lambda_edge_sdf > 0 and
                       args.edge_sdf_start <= iteration < args.edge_sdf_end)
    grad_tv_active = (args.lambda_gradient_tv > 0 and
                      args.gradient_tv_start <= iteration < args.gradient_tv_end)

    # [EDGE-SDF-LOSS] 只有 edge_sdf_active 时才计算全部内部顶点的 SDF
    if edge_sdf_active:
        num_internal = model.num_int_verts
        internal_verts = model.vertices[:num_internal]
        vert_sdf_internal = model.query_vert_sdf(internal_verts)
        edge_loss, edge_num_edges = edge_sdf_loss(
            model,
            vert_sdf=vert_sdf_internal,
            sdf_threshold=args.edge_sdf_threshold,
            edge_length_threshold=args.edge_length_threshold,
            target=args.edge_sdf_target
        )

    # [GRADIENT-TV-LOSS] 计算穿过表面的边（两端点 SDF 异号）的梯度变化
    if grad_tv_active:
        # 优先复用 Eikonal 已计算的梯度
        if eik_gradients is not None and eik_vertex_indices is not None:
            grad_tv_vertex_indices = eik_vertex_indices
            grad_tv_gradients = eik_gradients
        else:
            # 独立采样并计算梯度（使用所有内部顶点）
            grad_tv_points, grad_tv_vertex_indices, _, _ = sample_eikonal_points(
                model,
                num_samples=args.num_eikonal_samples,
                band=None,  # 使用所有内部顶点
                use_tet_sampling=False
            )
            if grad_tv_points.shape[0] > 0:
                _, grad_tv_gradients, _ = model.compute_sdf_gradients(
                    points=grad_tv_points,
                    mode=args.gradient_mode,
                    taps=args.gradient_taps,
                    eps=args.gradient_eps
                )
            else:
                grad_tv_gradients = None
                grad_tv_vertex_indices = None

        if grad_tv_gradients is not None and grad_tv_vertex_indices is not None:
            grad_tv_loss, grad_tv_num_edges = gradient_tv_loss_from_sampled(
                model,
                vertex_indices=grad_tv_vertex_indices,
                vert_gradients=grad_tv_gradients,
                surface_only=args.gradient_tv_surface_only,
                normalize_by_edge_length=args.gradient_tv_normalize
            )

    # [MESH-LOSS] Mesh 正则化 Loss (normal + depth + occupancy)
    mesh_loss = torch.tensor(0.0, device=device)
    mesh_loss_active = False
    mesh_has_any_weight = (args.lambda_mesh_normal > 0 or
                           args.lambda_mesh_depth > 0 or
                           args.lambda_mesh_occupancy > 0 or
                           args.lambda_mesh_occupancy_labels > 0)
    if mesh_has_any_weight and mesh_regularizer.should_apply(iteration):
        # 使用渲染的深度计算 mesh loss (MILo风格：支持期望/中位深度混合)
        gaussian_depth = render_pkg['depth']  # (H, W)
        gaussian_depth_med = render_pkg['depth_med']  # (H, W) 中位深度
        rendered_normal_for_mesh = render_pkg['normal']  # (3, H, W) 世界空间
        mesh_reg_result = mesh_regularizer(
            model=model,
            camera=camera,
            gaussian_depth=gaussian_depth,
            gaussian_depth_med=gaussian_depth_med,  # MILo: 中位深度用于混合
            gaussian_normal=rendered_normal_for_mesh,  # 渲染的法线（世界空间）
            normal_source=args.mesh_normal_source,     # "depth", "depth_med", or "rendered"
            spatial_lr_scale=spatial_lr_scale,      # MILo: 使用spatial_lr_scale作为depth scale
            iteration=iteration,
            gt_mask=gt_mask,  # 传入GT alpha mask，排除背景散乱法线
            train_cameras=train_cameras,
        )
        mesh_loss = mesh_reg_result['loss']
        mesh_loss_active = mesh_reg_result['active']

    # [DEPTH-NORMAL-LOSS] 深度推导法线与渲染法线一致性 Loss
    normal_depth_loss = torch.tensor(0.0, device=device)
    normal_depth_med_loss = torch.tensor(0.0, device=device)
    normal_depth_active = (
        (args.lambda_normal_depth > 0 or args.lambda_normal_depth_med > 0)
        and iteration >= args.normal_depth_start
        and iteration < args.normal_depth_end
    )
    if normal_depth_active:
        rendered_normal = render_pkg['normal']  # (3, H, W) 世界坐标系

        # [COORD-FIX] 将渲染法线从世界空间变换到相机空间（仿 svraster）
        # depth_to_normal 计算的法线在相机空间，需要统一坐标系
        # world_view_transform = W2V.T，所以 world_view_transform[:3,:3] = R.T
        # 法线变换（行向量）: n_view = n_world @ R.T = n_world @ world_view_transform[:3,:3]
        R = camera.world_view_transform[:3, :3].to(device)
        rendered_normal_view = rendered_normal.permute(1, 2, 0) @ R  # (H, W, 3) world-to-view
        rendered_normal_view = F.normalize(rendered_normal_view, dim=-1)  # 归一化
        rendered_normal_view = rendered_normal_view.permute(2, 0, 1)  # (3, H, W)

        # 构建mask：只在有效区域计算（alpha > 0.5 且 gt_mask）
        alpha_mask = render_pkg['alpha'] > 0.5  # (H, W)
        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask_2d = gt_mask.squeeze(0) > 0.5
            else:
                gt_mask_2d = gt_mask > 0.5
            valid_mask = alpha_mask & gt_mask_2d
        else:
            valid_mask = alpha_mask

        if args.lambda_normal_depth > 0:
            # 从期望深度推导法线（相机空间）
            depth_normal = depth_to_normal(camera, render_pkg['depth'])  # (3, H, W)
            normal_depth_loss = compute_normal_consistency_loss(
                rendered_normal_view, depth_normal, valid_mask
            )

        if args.lambda_normal_depth_med > 0:
            # 从中位深度推导法线（相机空间）
            depth_med_normal = depth_to_normal(camera, render_pkg['depth_med'])  # (3, H, W)
            normal_depth_med_loss = compute_normal_consistency_loss(
                rendered_normal_view, depth_med_normal, valid_mask
            )

    # [MASK-LOSS] Mask Loss: 约束 transmittance 与 GT mask 对齐
    mask_loss = torch.tensor(0.0, device=device)
    if args.lambda_mask > 0 and iteration >= args.mask_from and iteration <= args.mask_until:
        if gt_mask is not None and gt_mask.numel() > 0:
            # T = 1 - alpha，alpha 是累积不透明度
            T = 1.0 - render_pkg['alpha']  # (H, W)
            # GT mask: 物体=1, 背景=0 → 目标 T: 物体=0, 背景=1
            gt_mask_2d = gt_mask.squeeze() if gt_mask.dim() == 3 else gt_mask
            gt_T = 1.0 - gt_mask_2d  # (H, W)
            mask_loss = F.mse_loss(T, gt_T)

    # [T-CONCEN-LOSS] Transmittance Concentration Loss: 鼓励 T 二值化到 0 或 1
    T_concen_loss = torch.tensor(0.0, device=device)
    if args.lambda_T_concen > 0:
        T = 1.0 - render_pkg['alpha']  # (H, W)
        # prob_concen_loss: T² * (1-T)²，当 T=0 或 T=1 时为 0，T=0.5 时最大
        T_concen_loss = (T.square() * (1.0 - T).square()).mean()

    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           args.lambda_dist * dl_loss + \
           args.lambda_sh * render_pkg['sh_reg'] + \
           (args.lambda_eikonal * eik_loss if eikonal_active else 0.0) + \
           (args.lambda_curvature * curv_weight_decay * curv_loss if curvature_active else 0.0) + \
           mesh_loss + \
           (args.lambda_normal_depth * normal_depth_loss if normal_depth_active and args.lambda_normal_depth > 0 else 0.0) + \
           (args.lambda_normal_depth_med * normal_depth_med_loss if normal_depth_active and args.lambda_normal_depth_med > 0 else 0.0) + \
           args.lambda_mask * mask_loss + \
           args.lambda_T_concen * T_concen_loss + \
           (args.lambda_edge_sdf * edge_loss if edge_sdf_active else 0.0) + \
           (args.lambda_gradient_tv * grad_tv_loss if grad_tv_active else 0.0)

    loss.backward()

    # [DEBUG-MEMORY] backward 后
    if DEBUG_MEMORY:
        memory_monitor.log_delta(iteration, "after_backward", model)

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

    tet_optim.update_learning_rate(iteration)

    if do_sh_up:
        model.sh_up()

    # if iteration % 10 == 0:
    #     with torch.no_grad():
    #         render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
    #         sample_image = render_pkg['render']
    #         sample_image = sample_image.permute(1, 2, 0)
    #         sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
    #         sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
    #         video_writer.write(pad_image2even(sample_image))

    if do_cloning and not model.frozen:
        # [DEBUG-MEMORY] densification 前
        if DEBUG_MEMORY:
            memory_monitor.log(iteration, "before_densify", model, print_now=True)

        with torch.no_grad():
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]

            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)
            sample_image = render_pkg['render']
            sample_image = sample_image.permute(1, 2, 0)
            sample_image = (sample_image.detach().cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
            # video_writer.write(pad_image2even(sample_image))

            gc.collect()
            torch.cuda.empty_cache()

            # [DEBUG-MEMORY] collect_render_stats 前
            if DEBUG_MEMORY:
                memory_monitor.log(iteration, "before_collect_stats", model, print_now=True)

            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()

            # [DEBUG-MEMORY] collect_render_stats 后
            if DEBUG_MEMORY:
                memory_monitor.log_delta(iteration, "after_collect_stats", model)

            # target_addition = targets[dschedule.index(iteration)] - model.vertices.shape[0]
            target_addition = args.budget - model.vertices.shape[0]

            apply_densification(
                stats,
                model       = model,
                tet_optim   = tet_optim,
                args        = args,
                iteration   = iteration,
                device      = device,
                sample_cam  = sample_camera,
                sample_image= sample_image,     # whatever RGB debug frame you use
                target_addition= target_addition

            )
            del stats
            gc.collect()
            torch.cuda.empty_cache()

            # [DEBUG-MEMORY] densification 后
            if DEBUG_MEMORY:
                memory_monitor.log_delta(iteration, "after_densify", model)

    # 顶点剪枝：独立执行，需要单独收集 stats
    # prune 在 prune_start, prune_start+interval, prune_start+2*interval... 执行
    do_prune = (args.prune_enabled
                and iteration >= args.prune_start
                and iteration <= args.prune_end
                and (iteration - args.prune_start) % args.densify_interval == 0
                and not model.frozen)
    if do_prune:
        with torch.no_grad():
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]
            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()
            tet_optim.prune(
                peak_contrib=stats.peak_contrib,
                prune_contrib_threshold=args.prune_contrib_threshold,
                prune_sdf_far_threshold=args.prune_sdf_far_threshold
            )
            del stats
            gc.collect()
            torch.cuda.empty_cache()

    psnr = -20 * math.log10(math.sqrt(l2_loss.detach().cpu().clip(min=1e-6).item()))
    psnrs[-1].append(psnr)

    disp_ind = max(len(psnrs)-2, 0)
    avg_psnr = sum(psnrs[disp_ind]) / max(len(psnrs[disp_ind]), 1)

    # 获取 inv_s 的值（NeuS 的 SDF->density 锐度参数）
    inv_s_value = model.backbone.inv_s.item() if hasattr(model.backbone, 'inv_s') else 0.0

    postfix_dict = {
        "PSNR": repr(f"{psnr:>5.2f}"),
        "Mean": repr(f"{avg_psnr:>5.2f}"),
        "#V": len(model),
        "#T": model.indices.shape[0],
        "inv_s": repr(f"{inv_s_value:.1f}"),  # 添加 inv_s 输出
        "Eik": repr(f"{eik_loss.item():.3f}"),
        "Curv": repr(f"{curv_loss.item():.1e}"),
    }
    # Coarse-to-Fine: 显示当前激活层数
    if args.coarse2fine_enabled:
        postfix_dict["C2F"] = f"{model.backbone.active_levels}/{model.backbone.L}"
    # [GRADIENT-TV-LOSS] 显示 gradient tv loss（仅当激活时）
    if grad_tv_active:
        postfix_dict["GTV"] = repr(f"{grad_tv_loss.item():.3f}")
    # [MESH-LOSS] 显示 mesh loss（仅当激活时）
    if mesh_loss_active:
        postfix_dict["Mesh"] = repr(f"{mesh_loss.item():.3f}")
    # [DEPTH-NORMAL-LOSS] 显示 depth normal loss（仅当激活时）
    if args.lambda_normal_depth > 0:
        postfix_dict["ND"] = repr(f"{normal_depth_loss.item():.3f}")
    if args.lambda_normal_depth_med > 0:
        postfix_dict["NDm"] = repr(f"{normal_depth_med_loss.item():.3f}")
    progress_bar.set_postfix(postfix_dict)

    # 每k次迭代保存渲染图
    if args.save_render_interval > 0 and iteration % args.save_render_interval == 0:
        with torch.no_grad():
            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)

            # 1. RGB 渲染图
            rgb_img = render_pkg['render'].permute(1, 2, 0)
            rgb_img = (rgb_img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # 2. 期望深度图（归一化可视化）
            depth_map = render_pkg['depth']  # (H, W)
            depth_np = depth_map.detach().cpu().numpy()
            valid_mask = (depth_np > 0) & (depth_np < 1e6)
            if valid_mask.any():
                d_min, d_max = depth_np[valid_mask].min(), depth_np[valid_mask].max()
                depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
            else:
                depth_norm = np.zeros_like(depth_np)
            depth_norm = np.where(valid_mask, depth_norm, 0)
            depth_vis = (plt.cm.turbo(depth_norm)[:, :, :3] * 255).astype(np.uint8)

            # 3. 中位深度图（归一化可视化）
            depth_med_map = render_pkg['depth_med']  # (H, W)
            depth_med_np = depth_med_map.detach().cpu().numpy()
            valid_mask_med = (depth_med_np > 0) & (depth_med_np < 1e6)
            if valid_mask_med.any():
                d_min_med, d_max_med = depth_med_np[valid_mask_med].min(), depth_med_np[valid_mask_med].max()
                depth_med_norm = (depth_med_np - d_min_med) / (d_max_med - d_min_med + 1e-8)
            else:
                depth_med_norm = np.zeros_like(depth_med_np)
            depth_med_norm = np.where(valid_mask_med, depth_med_norm, 0)
            depth_med_vis = (plt.cm.turbo(depth_med_norm)[:, :, :3] * 255).astype(np.uint8)

            # 4. Gaussian 法线图（从渲染的 normal 输出）
            # [COORD-FIX] 转换到相机空间后可视化，与深度法线一致
            # MILo 可视化惯例: (1 - normal) / 2，而非 (normal + 1) / 2
            gauss_normal = render_pkg['normal'].permute(1, 2, 0)  # (H, W, 3) 世界空间
            R_vis = sample_camera.world_view_transform[:3, :3].to(gauss_normal.device)
            gauss_normal_view = gauss_normal @ R_vis  # world->view: world_view_transform 已经是 W2V.T
            gauss_normal_view = F.normalize(gauss_normal_view, dim=-1)
            gauss_normal_vis = ((1 - gauss_normal_view.detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

            # 5. 深度转法线图（期望深度）
            depth_normal = depth_to_normal(sample_camera, render_pkg['depth'])  # (3, H, W)
            depth_normal_vis = ((1 - depth_normal.permute(1, 2, 0).detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

            # 6. 深度转法线图（中位深度）
            depth_med_normal = depth_to_normal(sample_camera, render_pkg['depth_med'])  # (3, H, W)
            depth_med_normal_vis = ((1 - depth_med_normal.permute(1, 2, 0).detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

            # 整合成 2x3 网格图: RGB, Depth, DepthMed | Normal, DepthNormal, DepthMedNormal
            H, W = rgb_img.shape[:2]
            combined1 = np.zeros((H * 2, W * 3, 3), dtype=np.uint8)
            combined1[:H, :W] = rgb_img                        # 左上: RGB
            combined1[:H, W:2*W] = depth_vis                   # 中上: Depth
            combined1[:H, 2*W:] = depth_med_vis                # 右上: Depth Median
            combined1[H:, :W] = gauss_normal_vis               # 左下: Gaussian Normal
            combined1[H:, W:2*W] = depth_normal_vis            # 中下: Depth->Normal
            combined1[H:, 2*W:] = depth_med_normal_vis         # 右下: DepthMed->Normal

            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.6, 2
            cv2.putText(combined1, 'RGB', (10, 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined1, 'Depth', (W + 10, 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined1, 'Depth Med', (2*W + 10, 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined1, 'Normal', (10, H + 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined1, 'Depth->N', (W + 10, H + 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined1, 'DepthMed->N', (2*W + 10, H + 25), font, font_scale, (255, 255, 255), thickness)

            # 保存渲染图
            save_path = render_save_dir / f"iter_{iteration:06d}.png"
            imageio.imwrite(str(save_path), combined1)

            # 5. Mesh 法线图和深度图（如果 mesh regularizer 激活）
            if mesh_has_any_weight and mesh_regularizer.should_apply(iteration):
                sample_gt_mask = getattr(sample_camera, 'gt_alpha_mask', None)
                if sample_gt_mask is not None:
                    sample_gt_mask = sample_gt_mask.cuda()
                mesh_reg_vis = mesh_regularizer(
                    model=model,
                    camera=sample_camera,
                    gaussian_depth=render_pkg['depth'],
                    gaussian_depth_med=render_pkg['depth_med'],  # MILo: 中位深度
                    gaussian_normal=render_pkg['normal'],        # 渲染的法线（世界空间）
                    normal_source=args.mesh_normal_source,       # "depth", "depth_med", or "rendered"
                    spatial_lr_scale=spatial_lr_scale,           # MILo: spatial_lr_scale
                    iteration=iteration,
                    gt_mask=sample_gt_mask,
                    train_cameras=train_cameras,
                )
                if mesh_reg_vis['active'] and mesh_reg_vis['mesh_normal'] is not None:
                    # Mesh 法线图 (MILo 可视化惯例: fix_normal_map + (1 - normal) / 2)
                    mesh_normal = mesh_reg_vis['mesh_normal']  # (H, W, 3)
                    # 使用 fix_normal_map 确保法线指向相机
                    mesh_normal_fixed = fix_normal_map(sample_camera, mesh_normal, normal_in_view_space=True)
                    mesh_normal_vis = ((1 - mesh_normal_fixed.detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

                    # Mesh 深度图
                    mesh_depth = mesh_reg_vis['mesh_depth']  # (H, W)
                    mesh_depth_np = mesh_depth.detach().cpu().numpy()
                    valid_mask_mesh = (mesh_depth_np > 0) & (mesh_depth_np < 1e6)
                    if valid_mask_mesh.any():
                        md_min, md_max = mesh_depth_np[valid_mask_mesh].min(), mesh_depth_np[valid_mask_mesh].max()
                        mesh_depth_norm = (mesh_depth_np - md_min) / (md_max - md_min + 1e-8)
                    else:
                        mesh_depth_norm = np.zeros_like(mesh_depth_np)
                    mesh_depth_norm = np.where(valid_mask_mesh, mesh_depth_norm, 0)
                    mesh_depth_vis = (plt.cm.turbo(mesh_depth_norm)[:, :, :3] * 255).astype(np.uint8)

                    # 整合成 1x2 网格图: Mesh Normal, Mesh Depth
                    combined2 = np.zeros((H, W * 2, 3), dtype=np.uint8)
                    combined2[:, :W] = mesh_normal_vis      # 左: Mesh Normal
                    combined2[:, W:] = mesh_depth_vis       # 右: Mesh Depth

                    # 添加标签
                    cv2.putText(combined2, 'Mesh Normal', (10, 25), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(combined2, 'Mesh Depth', (W + 10, 25), font, font_scale, (255, 255, 255), thickness)

                    # 保存mesh渲染图
                    mesh_render_path = render_save_dir / f"mesh_{iteration:06d}.png"
                    imageio.imwrite(str(mesh_render_path), combined2)

    # 每k次迭代用MT提取mesh并保存
    if args.save_mesh_interval > 0 and iteration % args.save_mesh_interval == 0:
        with torch.no_grad():
            # 获取顶点SDF
            vert_sdf = model.query_vert_sdf()  # (N_verts,)
            # 使用Marching Tetrahedra提取mesh
            mesh_verts, mesh_faces = marching_tetrahedra(
                model.vertices, model.indices, vert_sdf,
                num_internal=model.num_int_verts  # 排除包含外部球面采样点的四面体
            )
            if mesh_faces.shape[0] > 0:
                # 创建Meshes对象并保存
                mesh = Meshes(mesh_verts, mesh_faces)
                mesh_path = mesh_save_dir / f"mesh_{iteration:06d}.ply"
                save_mesh_as_ply(str(mesh_path), mesh)
                print(f"[Mesh] Saved mesh at iter {iteration}: {mesh_verts.shape[0]} verts, {mesh_faces.shape[0]} faces")

                # 渲染mesh的深度和法线并保存
                from utils.mesh_extraction import render_mesh
                try:
                    render_output = render_mesh(
                        mesh, sample_camera,
                        return_depth=True,
                        return_normals=True,
                        use_opengl=True
                    )

                    # 1. Mesh深度图
                    mesh_depth = render_output['depth']  # (H, W)
                    mesh_depth_np = mesh_depth.detach().cpu().numpy()
                    valid_mask_mesh = (mesh_depth_np > 0) & (mesh_depth_np < 1e6)
                    if valid_mask_mesh.any():
                        md_min, md_max = mesh_depth_np[valid_mask_mesh].min(), mesh_depth_np[valid_mask_mesh].max()
                        mesh_depth_norm = (mesh_depth_np - md_min) / (md_max - md_min + 1e-8)
                    else:
                        mesh_depth_norm = np.zeros_like(mesh_depth_np)
                    mesh_depth_norm = np.where(valid_mask_mesh, mesh_depth_norm, 0)
                    mesh_depth_vis = (plt.cm.turbo(mesh_depth_norm)[:, :, :3] * 255).astype(np.uint8)

                    # 2. Mesh法线图（从世界空间转到相机空间后可视化）
                    mesh_normal_world = render_output['normals']  # (H, W, 3) 世界空间
                    R_mesh = sample_camera.world_view_transform[:3, :3].to(mesh_normal_world.device)
                    mesh_normal_view = mesh_normal_world @ R_mesh  # world->view
                    mesh_normal_view = F.normalize(mesh_normal_view, dim=-1)
                    mesh_normal_vis = ((1 - mesh_normal_view.detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

                    # 3. 深度转法线图
                    mesh_depth_normal = depth_to_normal(sample_camera, mesh_depth)  # (3, H, W)
                    mesh_depth_normal_vis = ((1 - mesh_depth_normal.permute(1, 2, 0).detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

                    # 整合成 1x3 网格图: Mesh Normal | Mesh Depth | Mesh Depth->Normal
                    H, W = mesh_normal_vis.shape[:2]
                    combined_mesh = np.zeros((H, W * 3, 3), dtype=np.uint8)
                    combined_mesh[:, :W] = mesh_normal_vis           # 左: Mesh Normal
                    combined_mesh[:, W:2*W] = mesh_depth_vis         # 中: Mesh Depth
                    combined_mesh[:, 2*W:] = mesh_depth_normal_vis   # 右: Mesh Depth->Normal

                    # 添加标签
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale, thickness = 0.6, 2
                    cv2.putText(combined_mesh, 'Mesh Normal', (10, 25), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(combined_mesh, 'Mesh Depth', (W + 10, 25), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(combined_mesh, 'Mesh D->N', (2*W + 10, 25), font, font_scale, (255, 255, 255), thickness)

                    # 保存mesh渲染图
                    mesh_render_path = mesh_save_dir / f"mesh_render_{iteration:06d}.png"
                    imageio.imwrite(str(mesh_render_path), combined_mesh)
                    print(f"[Mesh] Saved mesh renders at iter {iteration}")
                except Exception as e:
                    print(f"[Mesh] Failed to render mesh at iter {iteration}: {e}")

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
# video_writer.release()

torch.cuda.synchronize()
torch.cuda.empty_cache()

if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])
results = test_util.evaluate_and_save(model, splits, args.output_path, args.tile_size, min_t)

with (args.output_path / "results.json").open("w") as f:
    all_data = dict(
        psnr = avged_psnrs[-1] if len(avged_psnrs) > 0 else 0,
        **results
    )
    json.dump(all_data, f, cls=CustomEncoder)

# with torch.no_grad():
#     epath = cam_util.generate_cam_path(train_cameras, 400)
#     eimages = []
#     for camera in tqdm(epath):
#         render_pkg = render(camera, model, min_t=min_t, tile_size=args.tile_size)
#         image = render_pkg['render']
#         image = image.permute(1, 2, 0)
#         image = image.detach().cpu().numpy()
#         eimages.append(pad_image2even(image))
# mediapy.write_video(args.output_path / "rotating.mp4", eimages)

model.save2ply(args.output_path / "ckpt.ply")
sd = model.state_dict()
sd['indices'] = model.indices
sd['empty_indices'] = model.empty_indices
torch.save(sd, args.output_path / "ckpt.pth")

# ============================================================================
# [MESH-EVAL] Mesh 提取与评估（复用 MILo 评估代码）
# ============================================================================

def get_dtu_info(dataset_path):
    """从数据集路径提取 DTU 信息"""
    src_path = os.path.normpath(str(dataset_path))
    scan_folder = os.path.basename(src_path)
    dtu_root = os.path.dirname(src_path)
    scan_digits = ''.join([c for c in scan_folder if c.isdigit()])
    scan_id = int(scan_digits) if scan_digits else 37

    # 检查是否为 DTU 数据集
    obs_mask_path = os.path.join(dtu_root, 'ObsMask', f'ObsMask{scan_id}_10.mat')
    calibration_path = os.path.join(dtu_root, 'Calibration', 'cal18')
    is_dtu = os.path.exists(obs_mask_path) or os.path.exists(calibration_path)

    return dtu_root, scan_id, is_dtu


def get_tnt_info(dataset_path):
    """从数据集路径提取 TNT 信息"""
    from eval.tnt.config import SCENES_TAU_DICT

    src_path = os.path.normpath(str(dataset_path))
    scene_folder = os.path.basename(src_path)
    tnt_root = os.path.dirname(src_path)

    # 场景名称可能是目录名（如 Barn, Truck）
    scene_name = scene_folder

    # 检查是否为 TNT 数据集
    # 方法1: 场景名在 TNT 场景列表中
    # 方法2: 存在 GT 点云和 COLMAP log 文件

    # 检查两种可能的 GT 路径：
    # 1. {tnt_root}/{scene_name}.ply（标准路径）
    # 2. {tnt_root}/GT_TNT/{scene_name}/{scene_name}.ply（用户的路径结构）
    gt_ply = os.path.join(tnt_root, f"{scene_name}.ply")
    gt_ply_alt = os.path.join(tnt_root, "GT_TNT", scene_name, f"{scene_name}.ply")
    colmap_log = os.path.join(tnt_root, f"{scene_name}_COLMAP_SfM.log")
    colmap_log_alt = os.path.join(tnt_root, "GT_TNT", scene_name, f"{scene_name}_COLMAP_SfM.log")

    # 如果 GT 在 GT_TNT 子目录，更新 tnt_root
    if os.path.exists(gt_ply_alt) and os.path.exists(colmap_log_alt):
        tnt_root = os.path.join(tnt_root, "GT_TNT", scene_name)
        gt_ply = gt_ply_alt
        colmap_log = colmap_log_alt

    is_tnt = scene_name in SCENES_TAU_DICT or (os.path.exists(gt_ply) and os.path.exists(colmap_log))

    return tnt_root, scene_name, is_tnt


def post_process_mesh(verts, faces, num_cluster=50):
    """后处理：移除小的 cluster"""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
        triangle_clusters, cluster_n_triangles, _ = mesh_o3d.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        return verts, faces

    threshold = max(np.sort(cluster_n_triangles)[-num_cluster] if len(cluster_n_triangles) >= num_cluster else 0, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
    mesh_o3d.remove_triangles_by_mask(triangles_to_remove)
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d.remove_degenerate_triangles()

    return np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles)


# 获取 DTU 信息
dtu_root, scan_id, is_dtu = get_dtu_info(args.dataset_path)
if is_dtu:
    print(f"\n[EVAL] DTU dataset detected: scan {scan_id}")
    print(f"       DTU root: {dtu_root}")

# 获取 TNT 信息
tnt_root, scene_name, is_tnt = get_tnt_info(args.dataset_path)
if is_tnt:
    print(f"\n[EVAL] TNT dataset detected: {scene_name}")
    print(f"       TNT root: {tnt_root}")

if not is_dtu and not is_tnt:
    print(f"\n[EVAL] Not a DTU or TNT dataset, skipping mesh evaluation")

# --------------------------------------------------------------------------
# 路径1: MT 提取
# --------------------------------------------------------------------------
if args.eval_mt:
    print("\n" + "="*60)
    print("[MT] Extracting mesh with Marching Tetrahedra...")
    print("="*60)

    with torch.no_grad():
        vert_sdf = model.query_vert_sdf()
        mt_verts, mt_faces = marching_tetrahedra(
            model.vertices, model.indices, vert_sdf,
            num_internal=model.num_int_verts
        )

    if mt_faces.shape[0] > 0:
        verts_np = mt_verts.cpu().numpy()
        faces_np = mt_faces.cpu().numpy()
        print(f"[MT] Raw mesh: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # 后处理
        verts_np, faces_np = post_process_mesh(verts_np, faces_np, num_cluster=50)
        print(f"[MT] After post-process: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # 保存训练坐标系的 mesh
        mt_mesh_path = args.output_path / "mesh_mt.ply"
        mt_mesh = o3d.geometry.TriangleMesh()
        mt_mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        mt_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        o3d.io.write_triangle_mesh(str(mt_mesh_path), mt_mesh)
        print(f"[MT] Saved: {mt_mesh_path}")

        # DTU 评估
        if args.eval_dtu and is_dtu:
            from eval.dtu.evaluate import evaluate_dtu_mesh
            print(f"\n[MT-EVAL] Running DTU evaluation...")
            results = evaluate_dtu_mesh(
                mesh_path=str(mt_mesh_path),
                train_cameras=train_cameras,
                dtu_path=dtu_root,
                scan_id=scan_id,
                output_dir=str(args.output_path / 'eval_dtu_mt'),
                cull=True,
                dilation_radius=args.cull_dilation_radius,
                dataset_path=str(args.dataset_path),  # 用于 scale_mat
                model_output_path=str(args.output_path),  # 用于 PCA 变换 (transform.txt)
            )
            if results:
                print(f"[MT-EVAL] Results: d2s={results['mean_d2s']:.4f}, s2d={results['mean_s2d']:.4f}, overall={results['overall']:.4f}")

        # TNT 评估
        if args.eval_tnt and is_tnt:
            from eval.tnt.evaluate import evaluate_tnt_mesh
            print(f"\n[MT-EVAL] Running TNT evaluation...")
            results = evaluate_tnt_mesh(
                mesh_path=str(mt_mesh_path),
                train_cameras=train_cameras,
                tnt_path=tnt_root,
                scene_name=scene_name,
                output_dir=str(args.output_path / 'eval_tnt_mt'),
                model_output_path=str(args.output_path),  # 用于 PCA 变换 (transform.txt)
                cull=False,  # TNT 使用基于深度渲染的 cull，可选
                min_visible_views=args.tnt_min_visible_views,
            )
            if results:
                print(f"[MT-EVAL] Results: precision={results['precision']:.4f}, recall={results['recall']:.4f}, fscore={results['fscore']:.4f}")
    else:
        print("[MT] No faces extracted")

# --------------------------------------------------------------------------
# 路径2: TSDF Fusion 提取
# --------------------------------------------------------------------------
if args.extract_tsdf:
    print("\n" + "="*60)
    print("[TSDF] Extracting mesh with TSDF Fusion...")
    print("="*60)

    from scripts.extract_mesh_tsdf import TSDFExtractor, post_process_mesh as tsdf_post_process

    with torch.no_grad():
        tsdf_extractor = TSDFExtractor(
            model,
            depth_ratio=args.tsdf_depth_ratio,
            tile_size=args.tile_size,
            min_t=args.min_t
        )

        print("[TSDF] Rendering depth maps...")
        tsdf_extractor.reconstruction(train_cameras)

        print(f"[TSDF] Fusing (voxel={args.tsdf_voxel_size}, trunc={args.tsdf_depth_trunc})...")
        tsdf_mesh = tsdf_extractor.extract_mesh(
            voxel_size=args.tsdf_voxel_size,
            sdf_trunc=args.tsdf_sdf_trunc,
            depth_trunc=args.tsdf_depth_trunc
        )

        verts_np = np.asarray(tsdf_mesh.vertices)
        faces_np = np.asarray(tsdf_mesh.triangles)
        print(f"[TSDF] Raw mesh: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # 后处理
        tsdf_mesh = tsdf_post_process(tsdf_mesh, cluster_to_keep=args.tsdf_num_cluster)
        verts_np = np.asarray(tsdf_mesh.vertices)
        faces_np = np.asarray(tsdf_mesh.triangles)
        print(f"[TSDF] After post-process: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

        # 保存训练坐标系的 mesh
        tsdf_mesh_path = args.output_path / "mesh_tsdf.ply"
        o3d.io.write_triangle_mesh(str(tsdf_mesh_path), tsdf_mesh)
        print(f"[TSDF] Saved: {tsdf_mesh_path}")

        # DTU 评估
        if args.eval_dtu and is_dtu:
            from eval.dtu.evaluate import evaluate_dtu_mesh
            print(f"\n[TSDF-EVAL] Running DTU evaluation...")
            results = evaluate_dtu_mesh(
                mesh_path=str(tsdf_mesh_path),
                train_cameras=train_cameras,
                dtu_path=dtu_root,
                scan_id=scan_id,
                output_dir=str(args.output_path / 'eval_dtu_tsdf'),
                cull=True,
                dilation_radius=args.cull_dilation_radius,
                dataset_path=str(args.dataset_path),  # 用于 scale_mat
                model_output_path=str(args.output_path),  # 用于 PCA 变换 (transform.txt)
            )
            if results:
                print(f"[TSDF-EVAL] Results: d2s={results['mean_d2s']:.4f}, s2d={results['mean_s2d']:.4f}, overall={results['overall']:.4f}")

        # TNT 评估
        if args.eval_tnt and is_tnt:
            from eval.tnt.evaluate import evaluate_tnt_mesh
            print(f"\n[TSDF-EVAL] Running TNT evaluation...")
            results = evaluate_tnt_mesh(
                mesh_path=str(tsdf_mesh_path),
                train_cameras=train_cameras,
                tnt_path=tnt_root,
                scene_name=scene_name,
                output_dir=str(args.output_path / 'eval_tnt_tsdf'),
                model_output_path=str(args.output_path),  # 用于 PCA 变换 (transform.txt)
                cull=False,  # TNT 使用基于深度渲染的 cull，可选
                min_visible_views=args.tnt_min_visible_views,
            )
            if results:
                print(f"[TSDF-EVAL] Results: precision={results['precision']:.4f}, recall={results['recall']:.4f}, fscore={results['fscore']:.4f}")

# [DEBUG-MEMORY] 输出显存使用汇总
if DEBUG_MEMORY:
    memory_monitor.summary()

print("\n" + "="*60)
print("[DONE] Training and evaluation completed.")
print("="*60)
