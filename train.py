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
import matplotlib.pyplot as plt
from utils import cam_util
from utils.train_util import *
from utils.model_util import eikonal_loss, curvature_loss
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
from utils.lib_bilagrid import BilateralGrid, total_variation_loss, slice
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ChainedScheduler
import gc
from utils.densification import collect_render_stats, apply_densification, determine_cull_mask
from utils.mesh_regularization import MeshRegularizer
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
args.iterations = 30000
args.ckpt = ""
args.resolution = 1
args.render_train = False

# Light Settings
args.max_sh_deg = 3
args.sh_interval = 0
args.sh_step = 1

# iNGP Settings
args.use_tccn = False
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
args.lambda_eikonal = 0.1       # Eikonal loss 权重
args.lambda_curvature = 5e-4   # Curvature loss 权重
args.gradient_mode = "numerical"  # "analytical" 或 "numerical"
args.gradient_taps = 4            # 数值梯度采样点数：4 或 6
args.gradient_eps = 1e-3          # 数值梯度步长

# [MESH-NORMAL-LOSS] Mesh 法线一致性 Loss 设置（参考 MILo）
args.lambda_mesh_normal = 0.0    # mesh normal loss 权重
args.mesh_normal_start = 8001     # 开始使用 mesh normal loss 的迭代
args.mesh_normal_stop = 18000     # 停止使用 mesh normal loss 的迭代
args.mesh_sdf_threshold = 0.0     # mesh 提取的 SDF 阈值

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

args.ablate_gradient = False
args.ablate_circumsphere = True
args.ablate_downweighing = False

# 渲染图保存设置
args.save_render_interval = 500  # 每k次迭代保存渲染图，0表示不保存


args = Args.from_namespace(args.get_parser().parse_args())

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=args.eval, resolution=args.resolution)

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
min_t = args.min_t

tet_optim = TetOptimizer(model, **args.as_dict())

# [MESH-NORMAL-LOSS] 创建 Mesh 正则化器
mesh_regularizer = MeshRegularizer(
    start_iter=args.mesh_normal_start,
    stop_iter=args.mesh_normal_stop,
    normal_weight=args.lambda_mesh_normal,
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

progress_bar = tqdm(range(args.iterations))
torch.cuda.empty_cache()
for iteration in progress_bar:
    do_delaunay = iteration % args.delaunay_interval == 0 and iteration < args.freeze_start
    do_freeze = iteration == args.freeze_start
    do_cloning = iteration in dschedule
    do_sh_up = not args.sh_interval == 0 and iteration % args.sh_interval == 0 and iteration > 0
    do_sh_step = iteration % args.sh_step == 0

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
    if args.lambda_eikonal > 0 or args.lambda_curvature > 0:
        # 只在活跃顶点（属于未被裁剪四面体的顶点）上计算 SDF 梯度
        _, gradients, hessian_diag = model.compute_sdf_gradients(
            points=model.active_vertices,
            mode=args.gradient_mode,
            taps=args.gradient_taps,
            eps=args.gradient_eps
        )
        if args.lambda_eikonal > 0:
            eik_loss = eikonal_loss(gradients)
        if args.lambda_curvature > 0 and hessian_diag is not None:
            curv_loss = curvature_loss(hessian_diag)

    # [MESH-NORMAL-LOSS] Mesh Normal 一致性 Loss
    mesh_normal_loss = torch.tensor(0.0, device=device)
    mesh_normal_active = False
    if args.lambda_mesh_normal > 0 and mesh_regularizer.should_apply(iteration):
        # 使用渲染的深度计算 mesh normal loss
        gaussian_depth = render_pkg['depth']  # (H, W)
        mesh_reg_result = mesh_regularizer(
            model=model,
            camera=camera,
            gaussian_depth=gaussian_depth,
            iteration=iteration,
            gt_mask=gt_mask,  # 传入GT alpha mask，排除背景散乱法线
        )
        mesh_normal_loss = mesh_reg_result['loss']
        mesh_normal_active = mesh_reg_result['active']

    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           args.lambda_dist * dl_loss + \
           args.lambda_sh * render_pkg['sh_reg'] + \
           args.lambda_eikonal * eik_loss + \
           args.lambda_curvature * curv_loss + \
           mesh_normal_loss  # [MESH-NORMAL-LOSS] 已经乘过权重

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
    postfix_dict = {
        "PSNR": repr(f"{psnr:>5.2f}"),
        "Mean": repr(f"{avg_psnr:>5.2f}"),
        "#V": len(model),
        "#T": model.indices.shape[0],
        "Eik": repr(f"{eik_loss.item():.3f}"),
        "Curv": repr(f"{curv_loss.item():.1e}"),
    }
    # [MESH-NORMAL-LOSS] 显示 mesh normal loss（仅当激活时）
    if mesh_normal_active:
        postfix_dict["MeshN"] = repr(f"{mesh_normal_loss.item():.3f}")
    progress_bar.set_postfix(postfix_dict)

    # 每k次迭代保存渲染图
    if args.save_render_interval > 0 and iteration % args.save_render_interval == 0:
        with torch.no_grad():
            render_pkg = render(sample_camera, model, min_t=min_t, tile_size=args.tile_size)

            # 1. RGB 渲染图
            rgb_img = render_pkg['render'].permute(1, 2, 0)
            rgb_img = (rgb_img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # 2. 深度图（归一化可视化）
            depth_map = render_pkg['depth']  # (H, W)
            depth_np = depth_map.detach().cpu().numpy()
            # 排除无效深度（0或极大值）进行归一化
            valid_mask = (depth_np > 0) & (depth_np < 1e6)
            if valid_mask.any():
                d_min, d_max = depth_np[valid_mask].min(), depth_np[valid_mask].max()
                depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
            else:
                depth_norm = np.zeros_like(depth_np)
            depth_norm = np.where(valid_mask, depth_norm, 0)
            # 使用 turbo colormap
            depth_vis = (plt.cm.turbo(depth_norm)[:, :, :3] * 255).astype(np.uint8)

            # 3. Gaussian 法线图（从渲染的 normal 输出）
            gauss_normal = render_pkg['normal'].permute(1, 2, 0)  # (H, W, 3)
            gauss_normal_vis = ((gauss_normal.detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            # 4. Mesh 法线图（如果 mesh regularizer 激活）
            mesh_normal_vis = np.zeros_like(rgb_img)
            if args.lambda_mesh_normal > 0 and mesh_regularizer.should_apply(iteration):
                # 获取 sample_camera 的 gt_mask（如果有）
                sample_gt_mask = getattr(sample_camera, 'gt_alpha_mask', None)
                if sample_gt_mask is not None:
                    sample_gt_mask = sample_gt_mask.cuda()
                mesh_reg_vis = mesh_regularizer(
                    model=model,
                    camera=sample_camera,
                    gaussian_depth=render_pkg['depth'],
                    iteration=iteration,
                    gt_mask=sample_gt_mask,
                )
                if mesh_reg_vis['active'] and mesh_reg_vis['mesh_normal'] is not None:
                    mesh_normal = mesh_reg_vis['mesh_normal']  # (H, W, 3)
                    mesh_normal_vis = ((mesh_normal.detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            # 整合成 2x2 网格图
            H, W = rgb_img.shape[:2]
            combined = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
            combined[:H, :W] = rgb_img                    # 左上: RGB
            combined[:H, W:] = depth_vis                  # 右上: Depth
            combined[H:, :W] = gauss_normal_vis           # 左下: Gaussian Normal
            combined[H:, W:] = mesh_normal_vis            # 右下: Mesh Normal

            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.6, 2
            cv2.putText(combined, 'RGB', (10, 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined, 'Depth', (W + 10, 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined, 'Gauss Normal', (10, H + 25), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(combined, 'Mesh Normal', (W + 10, H + 25), font, font_scale, (255, 255, 255), thickness)

            # 保存整合图
            save_path = render_save_dir / f"iter_{iteration:06d}.png"
            imageio.imwrite(str(save_path), combined)

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
