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
from utils.model_util import eikonal_loss, curvature_loss, sample_eikonal_points
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
args.image_folder = "images_2"
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
args.use_contraction = False  # True: 使用 Mip-NeRF 360 空间收缩, False: 仅线性归一化 (推荐关闭以正确计算 eikonal)
args.encoding_lr = 3e-3
args.final_encoding_lr = 3e-4
args.network_lr = 1e-3
args.final_network_lr = 1e-4
args.hidden_dim = 64
args.scale_multi = 0.35 # chosen such that 96% of the distribution is within the sphere
args.log2_hashmap_size = 23
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

# [SDF-MODIFICATION] Eikonal 和 Curvature Loss 设置（参考 Neuralangelo）
args.lambda_eikonal = 0.0 #0.1       # Eikonal loss 权重
args.lambda_curvature = 0.0 #5e-4   # Curvature loss 权重
args.eikonal_start = 0            # 开始使用 eikonal loss 的迭代
args.eikonal_end = 30000          # 停止使用 eikonal loss 的迭代
args.curvature_start = 0         # 开始使用 curvature loss 的迭代
args.curvature_end = 30000        # 停止使用 curvature loss 的迭代
args.gradient_mode = "numerical"  # "analytical" 或 "numerical"
args.gradient_taps = 4            # 数值梯度采样点数：4 或 6
args.gradient_eps = None          # 数值梯度步长，None=自动计算 (scene_scaling/resolution，参考Neuralangelo)
args.num_eikonal_samples = 200000  # Eikonal loss 采样点数（仅在 band_threshold=None 时使用）
args.eikonal_band_threshold = 0.02 # SDF band 阈值：仅计算 |SDF| < threshold 的顶点的 loss
                                   # 设为 None 则计算所有内部顶点
args.eikonal_tet_sampling = True  # 是否在 SDF 异号四面体内额外采样（参考 Neuralangelo）
                                   # True: band内顶点 + 四面体内采样; False: 仅 band内顶点

# [MESH-NORMAL-LOSS] Mesh 法线一致性 Loss 设置（参考 MILo）
args.lambda_mesh_normal = 0.0   # mesh normal loss 权重
args.lambda_mesh_depth = 0.0     # mesh depth loss 权重
args.lambda_mesh_occupancy = 0.0 # surface occupancy loss 权重
args.mesh_depth_scale = None     # depth loss 的 scale 参数 (None则使用spatial_lr_scale)
args.mesh_depth_ratio = 1.0      # MILo: median深度的混合比例 (0.0=仅期望深度, 1.0=仅中位深度)
args.mesh_normal_start = 5001     # 开始使用 mesh normal loss 的迭代
args.mesh_normal_stop = 18000     # 停止使用 mesh normal loss 的迭代
args.mesh_sdf_threshold = 0.0     # mesh 提取的 SDF 阈值

# [DEPTH-NORMAL-LOSS] 深度推导法线与渲染法线一致性 Loss（参考 MILo）
args.lambda_normal_depth = 0.0      # 渲染的normal vs 期望深度推导的normal
args.lambda_normal_depth_med = 0.0  # 渲染的normal vs 中位深度推导的normal
args.normal_depth_start = 5001         # 开始使用 normal-depth loss 的迭代
args.normal_depth_end = 30000       # 停止使用 normal-depth loss 的迭代

# [MASK-LOSS] Mask Loss 和 Transmittance Concentration Loss（参考 SVRaster）
args.lambda_mask = 0.1           # mask loss 权重：约束 T 与 GT mask 对齐
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
args.within_thresh = 0.5
args.total_thresh = 2.0
args.clone_min_contrib = 0.003
args.split_min_contrib = 0.01

# [SDF-DENSIFY] SDF 导向的分裂策略
args.sdf_densify_enabled = False    # 是否启用 SDF 导向的分裂
args.sdf_densify_threshold = 0.02     # |SDF| < 此值认为接近表面
args.sdf_densify_radius_thresh = 0.05 # 外接圆半径 > 此值认为需要细分
args.sdf_densify_start = 5000         # 开始使用 SDF 分裂的迭代数
args.sdf_densify_weight = 0.3         # SDF 分裂在总预算中的权重 (0.3 = 30%)

args.lambda_ssim = 0.2
args.min_t = 0.4
args.sample_cam = 8
args.data_device = 'cpu'
# [SDF-MODIFICATION] sdf_eps: CDF 贡献阈值 ε
# 保留 CDF 贡献 > ε 的四面体，例如 0.01 表示保留贡献 > 1% 的
args.sdf_eps = 0.01
args.alpha_threshold = 0.1
args.contrib_threshold = 0.0
args.threshold_start = 4500
args.voxel_size = 0.01
args.sdf_sphere_radius_scale = 1.2  # SDF球形初始化：点云半径 × scale / scene_scaling
args.init_num = 0  # 初始点云最小数量，不足则通过扰动扩充

args.ablate_gradient = False
args.ablate_circumsphere = True
args.ablate_downweighing = False

# 渲染图保存设置
args.save_render_interval = 500  # 每k次迭代保存渲染图，0表示不保存
args.save_mesh_interval = 1000   # 每k次迭代用MT提取mesh并保存，0表示不保存

# [MESH-EVAL] DTU Mesh 评估设置（参考 svraster）
args.eval_dtu = True            # 是否进行 DTU 评估


args = Args.from_namespace(args.get_parser().parse_args())

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

# [GEOMETRIC-INIT] 验证几何初始化是否成功
# 在训练开始前检查 Eikonal loss 和 SDF 分布
init_mesh_path = args.output_path / "init_mesh.ply"
init_result = model.verify_geometric_init(save_mesh_path=init_mesh_path)
if not init_result['is_success']:
    print("[WARNING] Geometric initialization may have issues. Consider checking sdf_bias parameter.")

min_t = args.min_t

tet_optim = TetOptimizer(model, **args.as_dict())

# [MESH-LOSS] 创建 Mesh 正则化器
mesh_regularizer = MeshRegularizer(
    start_iter=args.mesh_normal_start,
    stop_iter=args.mesh_normal_stop,
    normal_weight=args.lambda_mesh_normal,
    depth_weight=args.lambda_mesh_depth,
    occupancy_weight=args.lambda_mesh_occupancy,
    depth_scale=args.mesh_depth_scale,
    depth_ratio=args.mesh_depth_ratio,  # MILo: 期望/中位深度混合比例
    sdf_threshold=args.mesh_sdf_threshold,
    use_opengl=True,
)

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

video_writer = cv2.VideoWriter(str(args.output_path / "training.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                               pad_hw2even(sample_camera.image_width, sample_camera.image_height))

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

    st = time.time()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)
    render_pkg = render(camera, model, scene_scaling=model.scene_scaling, ray_jitter=ray_jitter, **args.as_dict())
    image = render_pkg['render']

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

    # 检查迭代范围
    eikonal_active = args.lambda_eikonal > 0 and args.eikonal_start <= iteration < args.eikonal_end
    curvature_active = args.lambda_curvature > 0 and args.curvature_start <= iteration < args.curvature_end
    if eikonal_active or curvature_active:
        # [NEW-SAMPLING] 使用新的采样策略：band内顶点 + SDF异号四面体内采样（可选）
        # 参考 Neuralangelo 的采样策略
        sampled_points, num_band, num_tet_samples = sample_eikonal_points(
            model,
            num_samples=args.num_eikonal_samples,
            band=args.eikonal_band_threshold,  # None 表示使用所有内部顶点
            use_tet_sampling=args.eikonal_tet_sampling  # False 则仅使用 band 内顶点
        )
        eik_num_samples = sampled_points.shape[0]

        # 如果采样点为空，跳过 loss 计算
        if sampled_points.shape[0] > 0:
            # [Coarse2Fine] 使用当前激活层的 eps (参考 Neuralangelo)
            if args.coarse2fine_enabled and args.gradient_eps is None:
                # eps = 1/resolution * scene_scaling (转换到世界坐标)
                eps = model.backbone.get_normal_epsilon() * float(model.scene_scaling)
            else:
                eps = args.gradient_eps

            # 在采样点上计算 SDF 梯度
            _, gradients, hessian_diag = model.compute_sdf_gradients(
                points=sampled_points,
                mode=args.gradient_mode,
                taps=args.gradient_taps,
                eps=eps
            )
            if eikonal_active:
                eik_loss = eikonal_loss(gradients)
            if curvature_active and hessian_diag is not None:
                curv_loss = curvature_loss(hessian_diag)
                # [Coarse2Fine] curvature 权重衰减 (参考 Neuralangelo)
                if args.coarse2fine_enabled:
                    curv_weight_decay = model.backbone.get_curvature_weight_decay()

    # [MESH-LOSS] Mesh 正则化 Loss (normal + depth + occupancy)
    mesh_loss = torch.tensor(0.0, device=device)
    mesh_loss_active = False
    mesh_has_any_weight = (args.lambda_mesh_normal > 0 or
                           args.lambda_mesh_depth > 0 or
                           args.lambda_mesh_occupancy > 0)
    if mesh_has_any_weight and mesh_regularizer.should_apply(iteration):
        # 使用渲染的深度计算 mesh loss (MILo风格：支持期望/中位深度混合)
        gaussian_depth = render_pkg['depth']  # (H, W)
        gaussian_depth_med = render_pkg['depth_med']  # (H, W) 中位深度
        mesh_reg_result = mesh_regularizer(
            model=model,
            camera=camera,
            gaussian_depth=gaussian_depth,
            gaussian_depth_med=gaussian_depth_med,  # MILo: 中位深度用于混合
            spatial_lr_scale=spatial_lr_scale,      # MILo: 使用spatial_lr_scale作为depth scale
            iteration=iteration,
            gt_mask=gt_mask,  # 传入GT alpha mask，排除背景散乱法线
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

        # [COORD-FIX] 将渲染法线从世界空间变换到相机空间（仿 MILo）
        # depth_to_normal 计算的法线在相机空间，需要统一坐标系
        # world_view_transform 定义时做了 .transpose(0,1)，直接用即可（同 MILo）
        # 法线从 world 变换到 view: n_view = n_world @ R
        R = camera.world_view_transform[:3, :3].to(device)  # (3, 3) world-to-view rotation
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
           args.lambda_T_concen * T_concen_loss

    loss.backward()

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
            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()
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
            # tet_optim.prune(**args.as_dict())
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
    # [BAND-THRESHOLD] 显示 band 内采样点数（仅当 eikonal/curvature 激活时）
    if (eikonal_active or curvature_active) and eik_num_samples > 0:
        postfix_dict["#Eik"] = eik_num_samples
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
            gauss_normal_view = gauss_normal @ R_vis  # 转换到相机空间（同 MILo，不带 .T）
            gauss_normal_view = F.normalize(gauss_normal_view, dim=-1)
            # [FIX] 使用 fix_normal_map 确保法线指向相机（与 mesh normal 可视化对齐）
            gauss_normal_fixed = fix_normal_map(sample_camera, gauss_normal_view, normal_in_view_space=True)
            gauss_normal_vis = ((1 - gauss_normal_fixed.detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

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
                    spatial_lr_scale=spatial_lr_scale,           # MILo: spatial_lr_scale
                    iteration=iteration,
                    gt_mask=sample_gt_mask,
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
                model.vertices, model.indices, vert_sdf
            )
            if mesh_faces.shape[0] > 0:
                # [COORD-FIX] 将 mesh 顶点从训练坐标系变换回原始世界坐标系
                inv_transform = np.linalg.inv(scene_info.transform)
                verts_np = mesh_verts.cpu().numpy()
                verts_hom = np.hstack([verts_np, np.ones((verts_np.shape[0], 1))])
                verts_world = (inv_transform @ verts_hom.T).T[:, :3]
                mesh_verts_world = torch.from_numpy(verts_world).float().to(mesh_verts.device)

                # 保存世界坐标系的 mesh
                mesh_world = Meshes(mesh_verts_world, mesh_faces)
                mesh_path = mesh_save_dir / f"mesh_{iteration:06d}.ply"
                save_mesh_as_ply(str(mesh_path), mesh_world)
                print(f"[Mesh] Saved mesh at iter {iteration}: {mesh_verts.shape[0]} verts, {mesh_faces.shape[0]} faces")

                # 渲染使用训练坐标系的 mesh（因为相机在训练坐标系）
                mesh_train = Meshes(mesh_verts, mesh_faces)
                from utils.mesh_extraction import render_mesh
                try:
                    render_output = render_mesh(
                        mesh_train, sample_camera,
                        return_depth=True,
                        return_normals=True,
                        use_opengl=True
                    )

                    # 1. Mesh深度图 - 处理可能的额外维度 (1, H, W, 1) -> (H, W)
                    mesh_depth = render_output['depth'].squeeze()  # squeeze 去掉所有多余维度
                    if mesh_depth.dim() == 3:  # 如果还是 (H, W, 1)
                        mesh_depth = mesh_depth[..., 0]
                    mesh_depth_np = mesh_depth.detach().cpu().numpy()
                    valid_mask_mesh = (mesh_depth_np > 0) & (mesh_depth_np < 1e6)
                    if valid_mask_mesh.any():
                        md_min, md_max = mesh_depth_np[valid_mask_mesh].min(), mesh_depth_np[valid_mask_mesh].max()
                        mesh_depth_norm = (mesh_depth_np - md_min) / (md_max - md_min + 1e-8)
                    else:
                        mesh_depth_norm = np.zeros_like(mesh_depth_np)
                    mesh_depth_norm = np.where(valid_mask_mesh, mesh_depth_norm, 0)
                    mesh_depth_vis = (plt.cm.turbo(mesh_depth_norm)[:, :, :3] * 255).astype(np.uint8)

                    # 2. Mesh法线图 - 处理可能的额外维度 (1, H, W, 3) -> (H, W, 3)
                    mesh_normal = render_output['normals'].squeeze()  # squeeze 去掉 batch 维度
                    if mesh_normal.dim() == 4:  # 如果是 (1, H, W, 3)
                        mesh_normal = mesh_normal[0]
                    # [COORD-FIX] 变换到相机空间（与 mesh_regularizer 保持一致）
                    from utils.geometry_utils import transform_normal_world_to_view
                    mesh_normal_view = transform_normal_world_to_view(mesh_normal, sample_camera)
                    # 使用 fix_normal_map 确保法线指向相机
                    mesh_normal_fixed = fix_normal_map(sample_camera, mesh_normal_view, normal_in_view_space=True)
                    mesh_normal_vis = ((1 - mesh_normal_fixed.detach().cpu().numpy()) / 2 * 255).clip(0, 255).astype(np.uint8)

                    # 3. 深度转法线图
                    mesh_depth_2d = mesh_depth if mesh_depth.dim() == 2 else mesh_depth.squeeze()
                    mesh_depth_normal = depth_to_normal(sample_camera, mesh_depth_2d)  # (3, H, W)
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
video_writer.release()

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

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, min_t=min_t, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(pad_image2even(image))
mediapy.write_video(args.output_path / "rotating.mp4", eimages)

model.save2ply(args.output_path / "ckpt.ply")
sd = model.state_dict()
sd['indices'] = model.indices
sd['empty_indices'] = model.empty_indices
torch.save(sd, args.output_path / "ckpt.pth")

# [MESH-EVAL] 提取最终 mesh 并保存（参考 svraster）
import subprocess
with torch.no_grad():
    # 提取最终 mesh
    vert_sdf = model.query_vert_sdf()
    final_mesh_verts, final_mesh_faces = marching_tetrahedra(
        model.vertices, model.indices, vert_sdf
    )
    if final_mesh_faces.shape[0] > 0:
        # [COORD-FIX] 将 mesh 顶点从训练坐标系变换回原始世界坐标系
        # 训练时做了 PCA 变换: 原始坐标 → pca_transform → 训练坐标
        # 现在需要逆变换: 训练坐标 → inv(pca_transform) → 原始坐标
        inv_transform = np.linalg.inv(scene_info.transform)
        verts_np = final_mesh_verts.cpu().numpy()
        verts_hom = np.hstack([verts_np, np.ones((verts_np.shape[0], 1))])
        verts_world = (inv_transform @ verts_hom.T).T[:, :3]
        final_mesh_verts_world = torch.from_numpy(verts_world).float().to(final_mesh_verts.device)

        final_mesh = Meshes(final_mesh_verts_world, final_mesh_faces)
        final_mesh_path = args.output_path / "final_mesh.ply"
        save_mesh_as_ply(str(final_mesh_path), final_mesh)
        print(f"[Mesh] Saved final mesh: {final_mesh_verts.shape[0]} verts, {final_mesh_faces.shape[0]} faces (transformed to world coords)")

        # [DTU-EVAL] DTU 评估（参考 svraster）
        if args.eval_dtu:
            try:
                script_path = os.path.join(os.path.dirname(__file__), 'eval_dtu_vanilla', 'evaluate_single_scene.py')
                src_path = os.path.normpath(str(args.dataset_path))
                scan_folder = os.path.basename(src_path)
                # 从文件夹名提取 scan_id（例如 scan37 -> 37）
                scan_digits = ''.join([c for c in scan_folder if c.isdigit()])
                scan_id = scan_digits if scan_digits else '37'
                dataset_root = os.path.dirname(src_path)
                eval_outdir = os.path.join(str(args.output_path), 'eval_dtu_vanilla')
                os.makedirs(eval_outdir, exist_ok=True)
                cmd = [
                    sys.executable, script_path,
                    '--input_mesh', str(final_mesh_path),
                    '--scan_id', str(scan_id),
                    '--output_dir', eval_outdir,
                    '--mask_dir', dataset_root,
                    '--DTU', dataset_root,
                    # 默认不做 scale_mat 变换（mesh 已在世界坐标系）
                ]
                print('[EVAL-DTU] mesh eval cmd:', ' '.join(cmd))
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"[EVAL-DTU] eval failed: {e}")
    else:
        print("[Mesh] No faces extracted, skipping mesh evaluation")

