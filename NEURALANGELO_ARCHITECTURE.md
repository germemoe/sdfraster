# Neuralangelo 代码结构与流程说明（/neuralangelo）

## 总览
- **框架基座**：`imaginaire/`（数据/模型/训练器的抽象与通用工具）。
- **Neuralangelo 具体实现**：`projects/neuralangelo/`（数据加载、模型、训练器、脚本、配置）。
- **其他示例**：`projects/nerf/`（ingp/nerf 参考）。
- **入口与环境**：`train.py`（通用入口）、`neuralangelo.yaml`（conda 环境）、`requirements.txt`、`docker/`（两种镜像）。
- **数据预处理**：`DATA_PROCESSING.md` + `projects/neuralangelo/scripts/*.sh/.py`（COLMAP、ffmpeg、格式转换）。

## 训练与评估流程（管线）
1) **数据准备**（外部脚本）：视频 → ffmpeg 抽帧 → COLMAP 求相机与点云 → `convert_*_to_json.py` 生成 Instant-NGP 风格 transforms JSON，或根据数据集（DTU/TNT）调用对应脚本。参见 `DATA_PROCESSING.md`。
2) **运行入口**：`train.py`（根目录）  
   - 解析命令行（logdir、config、wandb 等），加载 YAML 配置（例如 `projects/neuralangelo/configs/base.yaml`/`dtu.yaml`/`tnt.yaml`），初始化分布式/日志（`imaginaire/trainers/base.py`）。
   - 根据 `config.model`/`config.data` 创建数据集、模型、训练器（见下方模块）。
3) **数据加载**：`projects/neuralangelo/data.py`  
   - 读取 transforms JSON（相机位姿、内参、裁剪框），加载图像，构造 batch（射线/像素/掩码等）；支持 train/val 子集、图像下采样。
4) **模型构建**：`projects/neuralangelo/model.py`  
   - 包含 SDF 编码（多分辨率 hash grid）、MLP、粗细网络、法线/几何正则、颜色/外观嵌入等；激活、损失（eikonal、RGB、正则）在此定义。
   - 工具：`projects/neuralangelo/utils/{mlp,modules,spherical_harmonics,misc}.py`。
5) **训练逻辑**：`projects/neuralangelo/trainer.py`  
   - 继承 `imaginaire/trainers/base.py`：前向渲染（球追踪 / alpha 组合）、损失聚合、日志/可视化（`imaginaire/trainers/utils`）、学习率/ema 管理（`imaginaire/models/utils/model_average.py`）。
   - 使用 `projects/neuralangelo/utils/mesh.py` 做 marching cubes 提取、稀疏体素更新等辅助。
6) **评估/提取**：`projects/neuralangelo/scripts/extract_mesh.py`  
   - 读取 checkpoint 与 config，体素化 + Marching Cubes 导出网格，可选纹理、LCC 过滤。`mesh.py` 支持块状评估（block_res）控制显存。
7) **日志与分布式**：`imaginaire/utils/distributed.py`、`imaginaire/trainers/utils/logging.py`（wandb/本地）、`imaginaire/trainers/utils/meters.py`（指标）；`imaginaire/utils/cudnn.py`、`gpu_affinity.py` 做性能设置。

## 目录与文件功能
### 根目录
- `train.py`：通用入口，解析 args，加载 config，构造 dataset/model/trainer，启动分布式训练。
- `neuralangelo.yaml`、`requirements.txt`：环境依赖；`docker/Dockerfile-*`：COLMAP 与主环境镜像。
- `DATA_PROCESSING.md`：数据预处理指南；`README.md`：总体使用方法。

### imaginaire/（基础框架）
- `config_base.yaml` / `config.py`：配置系统（YAML 合并、命令行 override）。
- `datasets/base.py` + `datasets/utils/{dataloader,get_dataloader,sampler}.py`：数据集基类、分布式采样、构造 PyTorch DataLoader。
- `models/base.py` + `models/utils/{init_weight,model_average}.py`：模型基类、初始化、EMA。
- `trainers/base.py` + `trainers/utils/{get_trainer,logging,meters}.py`：训练器基类，日志/指标/分布式封装。
- `utils/*`：GPU 亲和、cudnn 设置、随机种子、可视化、分布式工具等。

### projects/neuralangelo/（核心实现）
- `configs/*.yaml`：场景/数据集的默认配置（base/dtu/tnt 等）。
- `data.py`：读取 transforms JSON/图像，构造批次（射线、像素、掩码），处理裁剪框、分辨率、子采样；提供 train/val dataset。
- `model.py`：Neuralangelo 模型主体：  
  - SDF 编码（多层 hashgrid）、粗细网络（MLP）、颜色/法线估计、外观嵌入。  
  - 损失：RGB、eikonal、正则（曲率、SDF 平滑等）、可选 mask/深度。  
  - 前向接口接收射线/样本，输出颜色、深度、法线、loss 字典。
- `trainer.py`：训练循环具体实现：前向渲染、损失聚合、日志、可视化、权重保存/EMA、学习率调度。
- `utils/`：
  - `mlp.py` / `modules.py`：MLP/编码模块、激活、PosEnc 等。
  - `spherical_harmonics.py`：SH 辅助。
  - `mesh.py`：网格提取（体素化、marching cubes、纹理生成、LCC 过滤）、块状评估。
  - `misc.py`：边界框、射线生成等杂项。
- `scripts/`：预处理与工具：
  - `convert_data_to_json.py`、`convert_dtu_to_json.py`、`convert_tnt_to_json.py`：把 COLMAP/DTU/TNT 数据转为 transforms JSON。
  - `preprocess*.sh` / `run_colmap.sh` / `run_ffmpeg.sh`：自动化抽帧、COLMAP、json 生成。
  - `extract_mesh.py`：从 checkpoint 体素化 + marching cubes 导出网格，可选纹理、LCC。
  - `visualize_colmap.ipynb` / `visualize_mesh.ipynb` / `visualize_transforms.ipynb`：检查相机/网格。

### projects/nerf/（参考实现）
- `configs/*.yaml`：NeRF/INGP 示例配置。
- `datasets/*`、`models/*`、`trainers/*`、`utils/*`：标准 NeRF/INGP pipeline，可参考配置和实现风格。

### third_party/、assets/
- `third_party`：外部依赖或补丁（具体内容视子目录而定）。
- `assets/teaser.gif`：项目示意。

## 模型与渲染设计摘要（Neuralangelo）
- **表示**：分层 hashgrid 编码的 SDF（粗+细网络），结合可选外观嵌入；SDF→密度转换用 NeuS/volSDF 风格（在 `model.py` 内实现）。
- **渲染**：沿射线采样，多段 SDF→alpha 组合，输出颜色/深度/法线；颜色由 SDF 上的梯度/法线与 SH/外观特征调制。
- **损失**：RGB 重建、eikonal、曲率/平滑正则、mask/深度（可选），外观一致性（如启用外观 embeddings）。
- **训练**：多 GPU 支持，EMA、学习率调度，日志（wandb/本地），可从 checkpoint/resume。
- **网格提取**：体素化 SDF，marching cubes，选项：块大小（block_res）、总分辨率、保留最大连通分量、纹理化。

使用入口：
```bash
torchrun --nproc_per_node=GPUS train.py --logdir=... --config=projects/neuralangelo/configs/{base,dtu,tnt,...}.yaml [--wandb ...]
```
Mesh 提取：
```bash
torchrun --nproc_per_node=GPUS projects/neuralangelo/scripts/extract_mesh.py \
  --config=logs/.../config.yaml --checkpoint=logs/.../xxx.pt \
  --output_file=mesh.ply --resolution=2048 --block_res=128 [--textured --keep_lcc]
```
