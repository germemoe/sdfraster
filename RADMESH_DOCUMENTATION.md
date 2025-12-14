# RadMesh 项目详细文档

> **RadMesh (Radiance Meshes)** 是一个基于四面体表示的体积渲染重建系统，结合 Instant-NGP 哈希编码实现高质量的 3D 场景重建。

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [核心代码文件详解](#3-核心代码文件详解)
4. [Pipeline 流程](#4-pipeline-流程)
5. [模型架构设计](#5-模型架构设计)
6. [关键算法实现](#6-关键算法实现)
7. [配置参数详解](#7-配置参数详解)
8. [使用指南](#8-使用指南)

---

## 1. 项目概述

### 1.1 简介

RadMesh 是一个创新的 3D 场景重建系统，使用**四面体 (Tetrahedron)** 作为基本几何单元，结合 **Instant-NGP (iNGP)** 哈希编码技术实现高效的神经辐射场表示。

### 1.2 核心技术特点

| 特性 | 描述 |
|------|------|
| **四面体表示** | 使用 Delaunay 四面体化作为几何骨架，比点云/高斯球更结构化 |
| **iNGP 哈希编码** | 多分辨率哈希网格实现快速位置编码 |
| **多头输出** | 同时预测密度、颜色、梯度、球谐系数 |
| **动态 Densify/Cull** | 基于渲染贡献度自适应顶点管理 |
| **球谐光照 (SH)** | 支持 0-4 阶球谐函数表示视角相关效果 |
| **Slang/CUDA 加速** | 高性能光栅化和光线追踪 |

### 1.3 输入/输出

```
输入:
├── 多视图图像 (RGB)
├── 相机参数 (COLMAP 或 Blender 格式)
└── 初始点云 (可选，可从 COLMAP 获取)

输出:
├── 训练好的辐射场模型 (checkpoint)
├── 渲染图像/视频
└── 可导出的三角网格 (.ply)
```

### 1.4 代码统计

| 模块 | 文件数 | 主要功能 |
|------|--------|---------|
| 根目录脚本 | 8 | 训练、评估、转换、基准测试 |
| data/ | 5 | 数据加载、相机处理 |
| models/ | 3 | 核心模型定义 |
| utils/ | 25+ | 工具函数库 |
| slang/ | ~10 | GPU 着色器 |

---

## 2. 目录结构

```
radmesh/
│
├── train.py                    # 主训练脚本 (~370 行)
├── evaluate.py                 # 离线评估脚本 (~64 行)
├── evaluate_traced.py          # JIT traced 模型评估 (~200 行)
├── evaluate_traced_optix.py    # OptiX 加速评估 (~300 行)
├── render_fisheye.py           # 鱼眼投影渲染 (~150 行)
├── tile_benchmark.py           # 瓦片渲染基准测试 (~100 行)
├── convert.py                  # 数据集格式转换 (~150 行)
├── hyperparam.py               # 批量实验管理 (~350 行)
│
├── data/                       # 数据处理模块
│   ├── camera.py               # Camera 类定义 (~460 行)
│   ├── colmap_loader.py        # COLMAP 格式解析
│   ├── dataset_readers.py      # 数据集读取器 (~500 行)
│   ├── loader.py               # 数据加载入口 (~160 行)
│   └── types.py                # 数据结构定义 (~45 行)
│
├── models/                     # 模型定义模块
│   ├── ingp_color.py           # 核心模型实现 (~600 行)
│   ├── frozen.py               # 冻结模型 (~150 行)
│   └── base_model.py           # 模型基类 (~150 行)
│
├── utils/                      # 工具函数库
│   │
│   │  # === 编码和特征 ===
│   ├── hashgrid.py             # iNGP 哈希网格实现
│   ├── ingp_util.py            # 哈希网格参数计算
│   │
│   │  # === 渲染和激活 ===
│   ├── model_util.py           # 密度/RGB 激活、SH 评估
│   ├── train_util.py           # render() 函数、学习率调度
│   ├── eval_sh_py.py           # 球谐评估 (0-4 阶)
│   │
│   │  # === 几何和拓扑 ===
│   ├── topo_utils.py           # 四面体体积、外心计算
│   ├── mesh_util.py            # 网格提取、面邻接
│   ├── graphics_utils.py       # 矩阵/向量操作
│   │
│   │  # === 优化相关 ===
│   ├── densification.py        # Densify/Cull 策略
│   ├── optim.py                # 自定义 Adam 优化器
│   ├── contraction.py          # 空间压缩
│   │
│   │  # === 数值安全 ===
│   ├── safe_math.py            # 安全数学运算
│   ├── jax_math.py             # JAX 版本安全数学
│   │
│   │  # === 其他工具 ===
│   ├── args.py                 # 参数解析
│   ├── loss_utils.py           # 损失函数
│   ├── ssim.py                 # SSIM 计算
│   ├── camera_path.py          # 相机轨迹生成
│   └── cam_util.py             # 相机采样工具
│
├── slang/                      # Slang GPU 着色器
│   ├── rasterize.slang         # 光栅化
│   ├── integrate.slang         # 光线积分
│   └── ...
│
├── gDel3D/                     # Delaunay 四面体化库 (C++)
│
├── tracers/                    # OptiX 追踪器 (可选)
│
└── scripts/                    # 辅助脚本
    ├── extract_mesh.py         # 网格提取
    └── run_full.csv            # 批量实验配置
```

---

## 3. 核心代码文件详解

### 3.1 主训练脚本 - `train.py`

**代码行数**: ~370 行
**功能**: 完整的训练流程入口

#### 主要流程

```python
def main():
    # 1. 参数解析
    args = Args.from_cli()

    # 2. 数据加载
    train_cameras, test_cameras, scene_info = loader.load_dataset(...)
    sampler = SimpleSampler(train_cameras)

    # 3. 模型初始化
    model = Model.init_from_pcd(point_cloud, cameras, ...)
    tet_optim = TetOptimizer(model, ...)

    # 4. 训练循环
    for iteration in range(30000):
        # 4a. Delaunay 更新
        if do_delaunay:
            tet_optim.update_triangulation(...)

        # 4b. 冻结模型
        if iteration == freeze_start:
            model = freeze_model(model)

        # 4c. Densify/Cull
        if do_cloning:
            collect_render_stats(...)
            apply_densification(...)

        # 4d. 渲染和损失计算
        render_pkg = render(camera, model, ...)
        loss = l1_loss + ssim_loss + distortion_loss + regularization

        # 4e. 反向传播
        loss.backward()
        tet_optim.step()

        # 4f. SH 升级
        if do_sh_up:
            model.sh_up()

    # 5. 保存和评估
    save_checkpoint(...)
    evaluate_and_save(...)
```

#### 关键训练阶段

| 阶段 | 迭代范围 | 操作 |
|------|---------|------|
| 预热 | 0 - 2000 | 基础训练，不做 Densify |
| Densify | 2000 - 16000 | 每 500 迭代执行顶点优化 |
| 稳定 | 16000 - 18000 | 停止 Densify，继续训练 |
| 冻结 | 18000 - 30000 | 冻结拓扑，精细化参数 |

---

### 3.2 数据加载 - `data/loader.py`

**代码行数**: ~160 行
**功能**: 统一的数据加载接口

#### load_dataset 函数

```python
def load_dataset(source_path, images_folder, eval_split, ...):
    """
    加载数据集

    参数:
        source_path: 数据集根目录
        images_folder: 图像文件夹名 (如 'images_2')
        eval_split: 是否分割测试集

    返回:
        train_cameras: 训练相机列表
        test_cameras: 测试相机列表
        scene_info: 场景信息 (点云、边界等)
    """
```

#### 数据加载流程

```
load_dataset()
│
├── 检测数据格式
│   ├── sparse/ 目录存在 → COLMAP 格式
│   └── transforms_train.json 存在 → Blender 格式
│
├── sceneLoadTypeCallbacks[格式](...)
│   │
│   ├── COLMAP 格式:
│   │   ├── readColmapCameras() - 解析 cameras.txt, images.txt
│   │   ├── readColmapPoints() - 读取 3D 点云
│   │   └── 返回 SceneInfo
│   │
│   └── Blender 格式:
│       ├── 解析 transforms_train.json
│       ├── 加载图像和相机矩阵
│       └── 返回 SceneInfo
│
├── load_cameras() - 加载所有相机
│   └── load_cam() - 单个相机
│       ├── 加载图像 (PIL)
│       ├── 按分辨率缩放
│       └── 创建 Camera 对象
│
├── transform_cameras_pca()
│   ├── 计算相机位置 PCA
│   ├── 对齐主轴到 XYZ
│   └── 变换相机和点云坐标
│
└── 返回 (train_cameras, test_cameras, scene_info)
```

---

### 3.3 相机类 - `data/camera.py`

**代码行数**: ~460 行
**功能**: 相机参数管理和射线生成

#### Camera 类结构

```python
class Camera:
    """
    相机参数和操作

    属性:
        K: 内参矩阵 [3, 3]
        R: 旋转矩阵 [3, 3]
        T: 平移向量 [3]
        image: 图像张量 [C, H, W]
        alpha_mask: Alpha 掩码 (可选)

        width, height: 图像尺寸
        fov_x, fov_y: 视场角
        distortion: 畸变参数 (可选)
    """
```

#### 核心方法

| 方法 | 功能 |
|------|------|
| `to_dict()` | 返回着色器所需的相机字典 |
| `to_rays()` | 生成世界空间射线 (origin, direction) |
| `get_camera_space_directions()` | 获取相机空间射线方向 |
| `get_world_space_rays()` | 获取世界空间射线 |
| `project()` | 3D 点投影到图像平面 |
| `unproject()` | 图像坐标反投影到 3D |

#### 射线生成

```python
def to_rays(self):
    """
    生成所有像素的射线

    返回:
        origins: [H, W, 3] 射线起点 (相机中心)
        directions: [H, W, 3] 射线方向 (归一化)
    """
    # 1. 生成像素网格
    u, v = meshgrid(width, height)

    # 2. 像素坐标 → 相机空间方向
    dirs_cam = (K_inv @ [u, v, 1].T).T

    # 3. 相机空间 → 世界空间
    dirs_world = (R.T @ dirs_cam.T).T
    dirs_world = normalize(dirs_world)

    # 4. 射线起点 = 相机中心
    origins = broadcast(camera_center, [H, W, 3])

    return origins, dirs_world
```

---

### 3.4 核心模型 - `models/ingp_color.py`

**代码行数**: ~600 行
**功能**: RadMesh 核心模型实现

#### 3.4.1 iNGPDW 类 (iNGP Dense Warping)

```python
class iNGPDW(nn.Module):
    """
    Instant-NGP 风格的多头编码网络

    输入:
        x: 压缩后的 3D 坐标 [N, 3]
        cr: 外心半径比 [N, 1]

    输出:
        density: 密度 [N, 1]
        rgb: RGB 颜色 [N, 3]
        gradient: 梯度向量 [N, 3]
        sh: 球谐系数 [N, sh_dim]
    """
```

**网络结构**:

```
x (3D 坐标)
    │
    ▼
┌─────────────────────────────────────┐
│  HashEmbedder (哈希网格编码)         │
│  ├── n_levels: 8                    │
│  ├── base_resolution: 64            │
│  ├── per_level_scale: 2             │
│  └── 输出: [N, 64] (8层 × 8维)      │
└─────────────────────────────────────┘
    │
    │ + 下加权 (approx_erf)
    │
    ▼
┌─────────────────────────────────────┐
│  4 个 MLP 头 (并行)                  │
│                                     │
│  ┌─────────────┐  ┌─────────────┐  │
│  │ density_net │  │ color_net   │  │
│  │ 64→64→1    │  │ 64→64→3    │  │
│  └─────────────┘  └─────────────┘  │
│                                     │
│  ┌─────────────┐  ┌─────────────┐  │
│  │ gradient_net│  │ sh_net      │  │
│  │ 64→64→3    │  │ 64→64→sh_dim│  │
│  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
(density, rgb, gradient, sh)
```

#### 3.4.2 Model 类 (四面体场景模型)

```python
class Model(nn.Module):
    """
    基于四面体的辐射场模型

    属性:
        contracted_vertices: 可学习顶点坐标 [V, 3]
        ext_vertices: 外壳固定顶点 [E, 3]
        indices: 四面体索引 [T, 4]
        backbone: iNGPDW 网络

        center: 场景中心 [3]
        scene_scaling: 场景缩放因子
    """
```

**初始化流程 (init_from_pcd)**:

```
init_from_pcd(point_cloud, cameras, ...)
│
├── 1. 计算场景参数
│   ├── center = mean(camera_centers)
│   └── scaling = max_distance(cameras, center)
│
├── 2. 点云处理
│   ├── 转换为 Open3D 点云
│   ├── 体素下采样 (voxel_size)
│   └── 添加微小噪声 (1e-3)
│
├── 3. 外壳点生成
│   └── Fibonacci 球面采样 (~1000 点)
│
├── 4. Delaunay 四面体化
│   └── gdel3D.Del().compute()
│       输入: 点云 + 外壳点
│       输出: indices [T, 4]
│
└── 5. 初始化网络
    └── backbone = iNGPDW(...)
```

**核心方法**:

| 方法 | 功能 |
|------|------|
| `compute_batch_features()` | 批量计算四面体特征 |
| `get_cell_values()` | 获取所有四面体的激活值 |
| `update_triangulation()` | 重新 Delaunay 四面体化 |
| `sh_up()` | 升级 SH 阶数 |
| `save_ckpt()` / `load_ckpt()` | 保存/加载检查点 |

**compute_batch_features 流程**:

```python
def compute_batch_features(self, vertices, indices, start, end):
    """
    计算一批四面体的特征

    流程:
    1. 获取四面体顶点坐标
    2. 计算外心和外径
    3. 在外球内采样 (k_samples 个点)
    4. 应用空间压缩 contract_mean_std()
    5. 通过 iNGPDW 编码

    返回:
        circumcenters: 外心坐标 [B, 3]
        density: 密度 [B, 1]
        rgb: 颜色 [B, 3]
        gradient: 梯度 [B, 3]
        sh: 球谐系数 [B, sh_dim]
    """
```

#### 3.4.3 TetOptimizer 类

```python
class TetOptimizer:
    """
    三路优化器管理

    包含:
        encoder_optim: Adam - 哈希网格参数
        network_optim: Adam - MLP 参数
        vertex_optim: Adam - 顶点坐标
    """
```

**学习率调度**:

| 参数组 | 初始 LR | 最终 LR | 调度方式 |
|--------|---------|---------|---------|
| encoder | 3e-3 | 3e-4 | 指数衰减 |
| network | 1e-3 | 1e-4 | 指数衰减 |
| vertices | 1e-4 | 1e-6 | 指数衰减 |

**Spiking LR**: 可选的周期性学习率脉冲，帮助跳出局部最优。

---

### 3.5 冻结模型 - `models/frozen.py`

**代码行数**: ~150 行
**功能**: 固定拓扑的推理模型

```python
class FrozenTetModel(nn.Module):
    """
    冻结拓扑的四面体模型

    特点:
    - 不再进行 Delaunay 更新
    - 每个四面体直接存储参数 (不通过 iNGP 编码)

    参数:
        density: [T, 1] 密度值
        rgb: [T, 3] RGB 基色
        gradient: [T, 3, 3] 线性梯度矩阵
        sh: [T, sh_dim, 3] 球谐系数
    """
```

**用途**:
- 训练后期的精细化
- 模型导出和部署
- 移动端推理

---

### 3.6 渲染函数 - `utils/train_util.py`

**核心函数**: `render()`

```python
def render(camera, model, tile_size=16, min_t=0.1, ray_jitter=None, ...):
    """
    渲染单个视图

    参数:
        camera: Camera 对象
        model: Model 或 FrozenTetModel
        tile_size: 瓦片大小 (默认 16×16)
        min_t: 最小深度
        ray_jitter: 射线抖动 (可选)

    返回:
        render_pkg: dict
            - image: 渲染图像 [C, H, W]
            - alpha: Alpha 通道 [H, W]
            - distortion: 畸变损失图
            - depth: 深度图 (可选)
    """
```

**渲染流程**:

```
render(camera, model, ...)
│
├── 1. 创建渲染网格
│   └── RenderGrid: 将图像分解为瓦片
│
├── 2. 顶点和瓦片着色 (Slang 内核)
│   ├── 投影四面体到屏幕空间
│   ├── 按深度排序四面体
│   ├── 为每个瓦片确定四面体范围
│   └── 返回: sorted_tetra_idx, tile_ranges
│
├── 3. 计算四面体特征
│   └── model.get_cell_values()
│       └── density, rgb, gradient, sh 激活
│
├── 4. Alpha Blend 渲染 (CUDA autograd)
│   └── AlphaBlendTiledRender.apply()
│       ├── 遍历每个瓦片的每条射线
│       ├── 在四面体内沿射线积分
│       └── Alpha blend 累积颜色
│
└── 5. 返回渲染结果
    └── {image, alpha, distortion, ...}
```

---

### 3.7 Densify/Cull - `utils/densification.py`

**功能**: 自适应顶点管理

#### RenderStats 结构

```python
@dataclass
class RenderStats:
    within_var_rays: Tensor  # [T, 2, 6] 前两个高方差射线
    total_var_moments: Tensor  # [T, 3] 累积方差统计
    tet_moments: Tensor  # [T, 4] 累积贡献度矩
    peak_contrib: Tensor  # [T] 峰值贡献率
```

#### apply_densification 流程

```
apply_densification(model, stats, ...)
│
├── 1. 统计分析
│   ├── peak_contrib > clone_min_contrib (0.003) → 克隆候选
│   ├── peak_contrib > split_min_contrib (0.01) → 分裂候选
│   └── within_var > threshold → 高不确定性
│
├── 2. 顶点操作
│   │
│   ├── 克隆 (Clone)
│   │   └── 高贡献度四面体
│   │       └── 沿高方差方向添加新顶点
│   │
│   ├── 分裂 (Split)
│   │   └── 贡献度高且形状差的四面体
│   │       └── 在边中点添加新顶点
│   │
│   └── 裁剪 (Cull)
│       └── 贡献度低的顶点
│           └── 从 Delaunay 中移除
│
└── 3. 预算约束
    ├── 总顶点数 ≤ budget (2M)
    └── 阈值控制添加速率
```

---

### 3.8 哈希网格编码 - `utils/hashgrid.py`

```python
class HashEmbedderOptimized(nn.Module):
    """
    Instant-NGP 风格的多分辨率哈希网格

    参数:
        n_levels: 网格层数 (默认 8)
        base_resolution: 最粗分辨率 (默认 64)
        finest_resolution: 最细分辨率
        log2_hashmap_size: 哈希表大小 (默认 2^23)
        n_features_per_level: 每层特征维度 (默认 8)
    """
```

**编码流程**:

```python
def forward(self, x):
    """
    x: [N, 3] 归一化坐标 [0, 1]

    流程:
    1. 对每个分辨率层:
       a. 计算体素坐标和插值权重
       b. 获取 8 个角点的哈希索引
       c. 查表获取特征
       d. 三线性插值

    2. 拼接所有层的特征

    返回: [N, n_levels * n_features_per_level]
    """
```

---

### 3.9 几何工具 - `utils/topo_utils.py`

#### 外心计算

```python
def calculate_circumcenters_torch(vertices):
    """
    计算四面体的外心和外径

    输入: vertices [N, 4, 3]
    输出: circumcenters [N, 3], radii [N]

    原理: 外心到四个顶点的距离相等
    """
    a = vertices[..., 1, :] - vertices[..., 0, :]
    b = vertices[..., 2, :] - vertices[..., 0, :]
    c = vertices[..., 3, :] - vertices[..., 0, :]

    # 叉积
    cross_bc = torch.cross(b, c)
    cross_ca = torch.cross(c, a)
    cross_ab = torch.cross(a, b)

    # 标量三重积
    denominator = 2.0 * torch.sum(a * cross_bc, dim=-1)

    # 外心
    aa = torch.sum(a * a, dim=-1)
    bb = torch.sum(b * b, dim=-1)
    cc = torch.sum(c * c, dim=-1)

    relative_cc = (aa * cross_bc + bb * cross_ca + cc * cross_ab) / denominator
    circumcenters = vertices[..., 0, :] + relative_cc

    # 外径
    radii = torch.norm(a - relative_cc, dim=-1)

    return circumcenters, radii
```

#### 四面体体积

```python
def calculate_tet_volume(vertices):
    """
    计算四面体体积 (带符号)

    V = |det([a-d, b-d, c-d])| / 6
    """
    a, b, c, d = vertices[..., 0], vertices[..., 1], vertices[..., 2], vertices[..., 3]
    return torch.det(torch.stack([a-d, b-d, c-d], dim=-1)) / 6
```

---

### 3.10 安全数学 - `utils/safe_math.py`

**目的**: 防止数值不稳定导致的 NaN/Inf

```python
class _SafeDiv(torch.autograd.Function):
    """安全除法，避免除零"""

    @staticmethod
    def forward(ctx, n, d):
        d_safe = remove_zero(d)  # |d| < tiny → tiny
        result = torch.clamp(n / d_safe, min_val, max_val)
        out = torch.where(torch.abs(d) < tiny_val, 0, result)
        ctx.save_for_backward(n, d, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 自定义梯度，避免 NaN
        ...

# 类似实现:
# safe_log, safe_exp, safe_sqrt, safe_pow
```

---

## 4. Pipeline 流程

### 4.1 训练 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练 Pipeline                            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 数据准备                                                    │
│     ├── COLMAP 稀疏重建 或 Blender 导出                         │
│     ├── convert.py --resize 缩放图像                           │
│     └── 生成 transforms.json 或 sparse/ 目录                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 数据加载                                                    │
│     ├── loader.load_dataset()                                   │
│     │   ├── 解析相机参数                                        │
│     │   ├── 加载图像                                            │
│     │   └── PCA 对齐坐标系                                      │
│     └── 返回 train_cameras, test_cameras, scene_info           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 模型初始化                                                  │
│     ├── Model.init_from_pcd()                                   │
│     │   ├── 计算场景中心和缩放                                  │
│     │   ├── 点云下采样                                          │
│     │   ├── 生成外壳点 (Fibonacci 球面)                         │
│     │   └── Delaunay 四面体化                                   │
│     └── TetOptimizer 初始化 (三路 Adam)                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 训练循环 (30000 迭代)                                       │
│                                                                 │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  每 10 迭代: Delaunay 更新                           │    │
│     │  └── update_triangulation()                          │    │
│     └─────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  每 500 迭代: Densify/Cull                           │    │
│     │  ├── collect_render_stats()                          │    │
│     │  └── apply_densification()                           │    │
│     └─────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  每次迭代: 渲染和优化                                │    │
│     │  ├── 随机采样相机                                    │    │
│     │  ├── render() → 渲染图像                             │    │
│     │  ├── 计算损失: L1 + SSIM + distortion + reg          │    │
│     │  ├── loss.backward()                                 │    │
│     │  └── optimizer.step()                                │    │
│     └─────────────────────────────────────────────────────┘    │
│                          │                                      │
│                          ▼                                      │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  迭代 18000: 冻结模型                                │    │
│     │  └── freeze_model() → FrozenTetModel                 │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 输出                                                        │
│     ├── checkpoint (ckpt.pth)                                   │
│     ├── 点云导出 (ckpt.ply)                                     │
│     ├── 测试集评估 (PSNR, SSIM, LPIPS)                          │
│     └── 渲染视频 (旋转轨迹)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 渲染 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        渲染 Pipeline                            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  输入: Camera + Model                                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 瓦片划分                                                    │
│     └── 图像 → 16×16 瓦片网格                                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 四面体投影和排序 (Slang GPU)                                │
│     ├── 投影四面体到屏幕空间                                    │
│     ├── 计算四面体-瓦片交集                                     │
│     ├── 按深度排序                                              │
│     └── 生成 tile_ranges                                        │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 特征计算                                                    │
│     └── model.get_cell_values()                                 │
│         ├── 计算外心和外径                                      │
│         ├── 空间压缩 (contraction)                              │
│         ├── iNGP 编码                                           │
│         └── 激活: density, rgb, gradient, sh                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Alpha Blend 积分 (CUDA autograd)                            │
│     ├── 对每个瓦片的每条射线:                                   │
│     │   ├── 遍历排序后的四面体                                  │
│     │   ├── 射线-四面体求交                                     │
│     │   ├── 沿射线积分密度和颜色                                │
│     │   └── Alpha blend 累积                                    │
│     └── 输出: image [C, H, W]                                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出: 渲染图像 + Alpha + Depth + Distortion                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Densify/Cull Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Densify/Cull Pipeline                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 收集渲染统计 (多视图)                                       │
│     └── collect_render_stats()                                  │
│         ├── 渲染多个视图                                        │
│         ├── 记录每个四面体的贡献度                              │
│         └── 计算方差和峰值贡献                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 分析统计                                                    │
│     ├── peak_contrib > 0.003 → 克隆候选                         │
│     ├── peak_contrib > 0.01 → 分裂候选                          │
│     └── 低贡献度 → 裁剪候选                                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 执行顶点操作                                                │
│     │                                                           │
│     ├── Clone (克隆)                                            │
│     │   └── 沿高方差方向添加顶点                                │
│     │                                                           │
│     ├── Split (分裂)                                            │
│     │   └── 在边中点添加顶点                                    │
│     │                                                           │
│     └── Cull (裁剪)                                             │
│         └── 移除低贡献顶点                                      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 重新 Delaunay 四面体化                                      │
│     └── update_triangulation()                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 模型架构设计

### 5.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RadMesh Model                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   输入: 3D 点云 + 相机参数                                           │
│                    │                                                 │
│                    ▼                                                 │
│   ┌────────────────────────────────────────┐                        │
│   │      Delaunay 四面体化 (gDel3D)         │                        │
│   │  ├── 点云 + 外壳点                      │                        │
│   │  └── 输出: 四面体索引 [T, 4]            │                        │
│   └────────────────────────────────────────┘                        │
│                    │                                                 │
│                    ▼                                                 │
│   ┌────────────────────────────────────────┐                        │
│   │         四面体几何计算                   │                        │
│   │  ├── 外心 (circumcenter)               │                        │
│   │  ├── 外径 (circumradius)               │                        │
│   │  └── 采样点 (k_samples)                │                        │
│   └────────────────────────────────────────┘                        │
│                    │                                                 │
│                    ▼                                                 │
│   ┌────────────────────────────────────────┐                        │
│   │         空间压缩 (Contraction)          │                        │
│   │  └── Instant-NGP 风格的坐标变换         │                        │
│   └────────────────────────────────────────┘                        │
│                    │                                                 │
│                    ▼                                                 │
│   ┌────────────────────────────────────────┐                        │
│   │         iNGPDW 编码网络                  │                        │
│   │                                        │                        │
│   │   ┌──────────────────────────────┐    │                        │
│   │   │  HashEmbedder (8层×8维=64D)  │    │                        │
│   │   └──────────────────────────────┘    │                        │
│   │                │                       │                        │
│   │                ▼                       │                        │
│   │   ┌──────────────────────────────┐    │                        │
│   │   │     4 个 MLP 头 (并行)        │    │                        │
│   │   │  ┌────┐┌────┐┌────┐┌────┐   │    │                        │
│   │   │  │ρ  ││RGB ││∇  ││SH  │   │    │                        │
│   │   │  └────┘└────┘└────┘└────┘   │    │                        │
│   │   └──────────────────────────────┘    │                        │
│   └────────────────────────────────────────┘                        │
│                    │                                                 │
│          ┌────────┼────────┬────────┐                               │
│          ▼        ▼        ▼        ▼                               │
│   ┌──────────┐┌──────────┐┌──────────┐┌──────────┐                 │
│   │ density  ││   rgb    ││ gradient ││    sh    │                 │
│   │  [T,1]   ││  [T,3]   ││  [T,3]   ││[T,sh_dim]│                 │
│   └──────────┘└──────────┘└──────────┘└──────────┘                 │
│                    │                                                 │
│                    ▼                                                 │
│   ┌────────────────────────────────────────┐                        │
│   │         激活和后处理                     │                        │
│   │  ├── density: softplus                 │                        │
│   │  ├── rgb: sigmoid                      │                        │
│   │  ├── gradient: offset_normalize        │                        │
│   │  └── sh: eval_sh (视角相关)            │                        │
│   └────────────────────────────────────────┘                        │
│                    │                                                 │
│                    ▼                                                 │
│   ┌────────────────────────────────────────┐                        │
│   │      Alpha Blend 体积渲染               │                        │
│   │  ├── 射线-四面体求交                    │                        │
│   │  ├── 沿射线积分                         │                        │
│   │  └── Alpha blend 累积                  │                        │
│   └────────────────────────────────────────┘                        │
│                    │                                                 │
│                    ▼                                                 │
│              输出: 渲染图像                                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 哈希网格编码架构

```
┌─────────────────────────────────────────────────────────────────┐
│              多分辨率哈希网格编码 (HashEmbedder)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   xyz 输入 [N, 3]                                               │
│       │                                                         │
│       ├──→ Level 0: 分辨率 64      ──→ 8D 特征                  │
│       ├──→ Level 1: 分辨率 128     ──→ 8D 特征                  │
│       ├──→ Level 2: 分辨率 256     ──→ 8D 特征                  │
│       ├──→ Level 3: 分辨率 512     ──→ 8D 特征                  │
│       ├──→ Level 4: 分辨率 1024    ──→ 8D 特征                  │
│       ├──→ Level 5: 分辨率 2048    ──→ 8D 特征                  │
│       ├──→ Level 6: 分辨率 4096    ──→ 8D 特征                  │
│       └──→ Level 7: 分辨率 8192    ──→ 8D 特征                  │
│                                                                 │
│       └──→ 拼接: 8 × 8 = 64D                                    │
│                                                                 │
│   哈希表参数:                                                   │
│   ├── log2_hashmap_size: 23 (8M 条目)                          │
│   ├── n_features_per_level: 8                                   │
│   ├── base_resolution: 64                                       │
│   └── per_level_scale: 2                                        │
│                                                                 │
│   编码过程:                                                     │
│   1. 计算体素坐标 (floor)                                       │
│   2. 计算插值权重 (frac)                                        │
│   3. 8 个角点哈希索引                                           │
│   4. 查表获取特征                                               │
│   5. 三线性插值                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 MLP 网络结构

```
┌─────────────────────────────────────────────────────────────────┐
│                    iNGPDW 网络结构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入: hash_features [N, 64]                                   │
│              │                                                  │
│              │ (下加权后)                                       │
│              │                                                  │
│   ┌──────────┼──────────┬──────────┬──────────┐                │
│   │          │          │          │          │                │
│   ▼          ▼          ▼          ▼          │                │
│                                                                 │
│  density_net    color_net     gradient_net    sh_net           │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐         │
│  │Linear  │    │Linear  │    │Linear  │    │Linear  │         │
│  │64→64   │    │64→64   │    │64→64   │    │64→64   │         │
│  │ReLU    │    │ReLU    │    │ReLU    │    │ReLU    │         │
│  ├────────┤    ├────────┤    ├────────┤    ├────────┤         │
│  │Linear  │    │Linear  │    │Linear  │    │Linear  │         │
│  │64→64   │    │64→64   │    │64→64   │    │64→64   │         │
│  │ReLU    │    │ReLU    │    │ReLU    │    │ReLU    │         │
│  ├────────┤    ├────────┤    ├────────┤    ├────────┤         │
│  │Linear  │    │Linear  │    │Linear  │    │Linear  │         │
│  │64→1    │    │64→3    │    │64→3    │    │64→sh   │         │
│  └────────┘    └────────┘    └────────┘    └────────┘         │
│       │             │             │             │              │
│       ▼             ▼             ▼             ▼              │
│   density[1]    rgb[3]      gradient[3]    sh[sh_dim]         │
│                                                                 │
│  sh_dim = 3 * (max_sh_deg + 1)^2                               │
│  例: max_sh_deg=3 → sh_dim = 3 * 16 = 48                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 球谐函数表示

```
┌─────────────────────────────────────────────────────────────────┐
│                    球谐函数 (Spherical Harmonics)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   用途: 表示视角相关的光照效果                                  │
│                                                                 │
│   SH 阶数和系数数量:                                            │
│   ┌───────┬────────────┬────────────┐                          │
│   │ 阶数  │ 基函数数量  │ RGB 总维度 │                          │
│   ├───────┼────────────┼────────────┤                          │
│   │ L=0   │ 1          │ 3          │                          │
│   │ L=1   │ 4          │ 12         │                          │
│   │ L=2   │ 9          │ 27         │                          │
│   │ L=3   │ 16         │ 48         │                          │
│   │ L=4   │ 25         │ 75         │                          │
│   └───────┴────────────┴────────────┘                          │
│                                                                 │
│   评估 (eval_sh):                                               │
│   输入: sh_coeffs [T, (L+1)², 3], view_dir [T, 3]              │
│   输出: color_offset [T, 3]                                     │
│                                                                 │
│   final_color = base_rgb + eval_sh(sh_coeffs, view_dir)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键算法实现

### 6.1 Delaunay 四面体化

```python
# 使用 gDel3D 库进行 Delaunay 四面体化
import gdel3D

def delaunay_tetrahedralization(points):
    """
    输入: points [N, 3] - 3D 点云
    输出: tetra_indices [T, 4] - 四面体顶点索引
    """
    delaunay = gdel3D.Del()
    tetra_indices = delaunay.compute(points)
    return tetra_indices
```

### 6.2 外心和外径计算

```python
def calculate_circumcenters_torch(vertices):
    """
    计算四面体的外心和外径

    输入: vertices [N, 4, 3] - 四面体顶点
    输出: circumcenters [N, 3], radii [N]
    """
    # 边向量
    a = vertices[..., 1, :] - vertices[..., 0, :]
    b = vertices[..., 2, :] - vertices[..., 0, :]
    c = vertices[..., 3, :] - vertices[..., 0, :]

    # 叉积
    cross_bc = torch.cross(b, c, dim=-1)
    cross_ca = torch.cross(c, a, dim=-1)
    cross_ab = torch.cross(a, b, dim=-1)

    # 分母 (6 * 体积)
    denominator = 2.0 * torch.sum(a * cross_bc, dim=-1, keepdim=True)

    # 边长平方
    aa = torch.sum(a * a, dim=-1, keepdim=True)
    bb = torch.sum(b * b, dim=-1, keepdim=True)
    cc = torch.sum(c * c, dim=-1, keepdim=True)

    # 相对外心
    numerator = aa * cross_bc + bb * cross_ca + cc * cross_ab
    relative_cc = numerator / denominator

    # 绝对外心
    circumcenters = vertices[..., 0, :] + relative_cc

    # 外径
    radii = torch.norm(relative_cc, dim=-1)

    return circumcenters, radii
```

### 6.3 空间压缩 (Contraction)

```python
def contract_mean_std(x, mean, std):
    """
    Instant-NGP 风格的空间压缩

    将无限空间映射到有限范围

    输入: x [N, 3] - 原始坐标
    输出: x_contracted [N, 3] - 压缩后坐标
    """
    # 归一化
    x_normalized = (x - mean) / std

    # 计算范数
    norm = torch.norm(x_normalized, dim=-1, keepdim=True)

    # 分段压缩
    # |x| <= 1: 保持不变
    # |x| > 1: 压缩到 [1, 2] 范围
    x_contracted = torch.where(
        norm <= 1,
        x_normalized,
        (2 - 1 / norm) * (x_normalized / norm)
    )

    return x_contracted
```

### 6.4 Alpha Blend 体积渲染

```python
def alpha_blend_integrate(ray_o, ray_d, tets, features):
    """
    沿射线进行 Alpha Blend 积分

    输入:
        ray_o: [H, W, 3] 射线起点
        ray_d: [H, W, 3] 射线方向
        tets: [T, 4, 3] 四面体顶点
        features: dict 包含 density, rgb, gradient, sh

    输出:
        color: [H, W, 3] 渲染颜色
        alpha: [H, W] 累积透明度
    """
    # 对每条射线:
    # 1. 与排序后的四面体求交
    # 2. 计算入点和出点
    # 3. 沿交线积分密度: alpha = 1 - exp(-density * length)
    # 4. Alpha blend 累积颜色

    T = 1.0  # 累积透射率
    color = 0.0

    for tet in sorted_tets:
        t_in, t_out = ray_tet_intersection(ray_o, ray_d, tet)
        if t_in < t_out:
            length = t_out - t_in
            alpha = 1 - exp(-density * length)

            # 沿交线采样颜色
            tet_color = sample_color(tet, features)

            # Alpha blend
            color = color + T * alpha * tet_color
            T = T * (1 - alpha)

            if T < 1e-4:
                break

    return color, 1 - T
```

### 6.5 PCA 坐标对齐

```python
def transform_poses_pca(poses):
    """
    将相机位置的 PCA 主轴对齐到 XYZ 坐标系

    输入: poses [N, 4, 4] - 相机位姿矩阵
    输出: poses_aligned [N, 4, 4], transform [4, 4]
    """
    # 提取相机中心
    t = poses[:, :3, 3]  # [N, 3]
    t_mean = t.mean(axis=0)
    t_centered = t - t_mean

    # PCA
    cov = t_centered.T @ t_centered
    eigval, eigvec = np.linalg.eig(cov)

    # 按特征值排序
    inds = np.argsort(eigval)[::-1]
    rot = eigvec[:, inds].T

    # 确保右手坐标系
    if np.linalg.det(rot) < 0:
        rot = np.diag([1, 1, -1]) @ rot

    # 构建变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = rot @ (-t_mean)

    # 应用变换
    poses_aligned = transform @ poses

    return poses_aligned, transform
```

---

## 7. 配置参数详解

### 7.1 iNGP 编码参数

```python
# 哈希网格配置
log2_hashmap_size = 23      # 哈希表大小: 2^23 = 8M 条目
base_resolution = 64        # 最粗分辨率
per_level_scale = 2         # 层间分辨率倍数
n_levels = 8                # 哈希网格层数
hashmap_dim = 8             # 每层特征维度

# 总特征维度: 8 * 8 = 64
```

### 7.2 MLP 配置

```python
# 网络结构
hidden_dim = 64             # 隐藏层维度

# 4 个输出头
density_net: (64, 64) → 1   # 密度
color_net: (64, 64) → 3     # RGB
gradient_net: (64, 64) → 3  # 梯度
sh_net: (64, 64) → sh_dim   # 球谐系数
```

### 7.3 学习率配置

```python
# 初始学习率
encoding_lr = 3e-3          # 哈希网格
network_lr = 1e-3           # MLP
vertices_lr = 1e-4          # 顶点坐标

# 最终学习率 (指数衰减)
encoding_lr_final = 3e-4
network_lr_final = 1e-4
vertices_lr_final = 1e-6

# Spiking LR (可选)
spike_duration = 500        # 脉冲周期
```

### 7.4 训练参数

```python
# 训练迭代
iterations = 30000

# Densify/Cull 参数
densify_start = 2000        # 开始迭代
densify_end = 16000         # 结束迭代
densify_interval = 500      # 间隔

# 冻结参数
freeze_start = 18000        # 冻结迭代

# Delaunay 更新
delaunay_interval = 10      # 每 N 迭代更新

# 顶点预算
budget = 2_000_000          # 最大顶点数
```

### 7.5 Densify 阈值

```python
# 贡献度阈值
clone_min_contrib = 0.003   # 克隆最小贡献
split_min_contrib = 0.01    # 分裂最小贡献

# 方差阈值
within_thresh = 0.5         # 高方差阈值
total_thresh = 2.0          # 总体阈值
```

### 7.6 球谐函数配置

```python
# SH 阶数
max_sh_deg = 3              # 最高 3 阶 (16 个基函数)

# SH 升级
sh_interval = 0             # 0 = 固定, >0 = 每 N 迭代升级
```

### 7.7 渲染参数

```python
# 瓦片大小
tile_size = 16              # 16×16 像素

# 深度范围
min_t = 0.1                 # 最小深度

# 采样
k_samples = 1               # 外球内采样数
```

### 7.8 损失权重

```python
# 主损失
lambda_l1 = 0.8             # L1 损失权重
lambda_ssim = 0.2           # SSIM 损失权重

# 正则化
lambda_distortion = 0.01    # 畸变损失
lambda_sh_reg = 0.001       # SH 正则化
weight_decay = 1e-5         # 权重衰减
```

---

## 8. 使用指南

### 8.1 环境配置

```bash
# 创建环境
conda create -n radmesh python=3.10
conda activate radmesh

# 安装依赖
pip install torch torchvision
pip install numpy scipy open3d plyfile imageio
pip install fused-ssim  # CUDA SSIM

# 安装 gDel3D (Delaunay)
cd gDel3D && pip install .

# 可选: tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### 8.2 数据准备

#### COLMAP 格式

```bash
# 运行 COLMAP 稀疏重建
colmap automatic_reconstructor \
    --workspace_path $DATASET \
    --image_path $DATASET/images

# 转换和缩放
python convert.py -s $DATASET --resize
```

#### Blender 格式

确保目录包含:
```
dataset/
├── transforms_train.json
├── transforms_test.json
└── images/
    ├── 000.png
    ├── 001.png
    └── ...
```

### 8.3 训练

```bash
# 基本训练
python train.py \
    --dataset_path /path/to/dataset \
    --output_path output/experiment \
    --iterations 30000

# 完整参数
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
    --dataset_path /data/bonsai \
    --image_folder images_2 \
    --output_path output/bonsai \
    --iterations 30000 \
    --max_sh_deg 3 \
    --log2_hashmap_size 23 \
    --densify_start 2000 \
    --densify_end 16000 \
    --freeze_start 18000
```

### 8.4 评估

```bash
# 基本评估
python evaluate.py \
    --output_path output/experiment \
    --dataset_path /path/to/dataset

# JIT traced 模型评估
python evaluate_traced.py \
    --output_path output/experiment \
    --dataset_path /path/to/dataset

# OptiX 加速评估
python evaluate_traced_optix.py \
    --output_path output/experiment \
    --dataset_path /path/to/dataset
```

### 8.5 网格提取

```bash
python scripts/extract_mesh.py \
    --output_path output/experiment \
    --dataset_path /path/to/dataset \
    --contrib_threshold 0.45
```

### 8.6 批量实验

```bash
# 使用 CSV 配置批量实验
python hyperparam.py \
    --queue_csv scripts/run_full.csv \
    --output_csv results/run_full.csv \
    --gpus 0,1,2,3
```

### 8.7 常见问题

**Q: 内存不足?**
- 减少 `log2_hashmap_size` (如 21)
- 减少 `budget` (如 1M)
- 使用更小的图像分辨率

**Q: 训练不收敛?**
- 检查相机参数是否正确
- 确保点云覆盖场景
- 调整学习率

**Q: 渲染质量差?**
- 增加训练迭代次数
- 提高 `max_sh_deg`
- 调整 Densify 参数

---

## 参考文献

1. Müller, T., et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." SIGGRAPH 2022.

2. Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.

3. Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." ECCV 2020.

---

*文档生成时间: 2025-12-10*
