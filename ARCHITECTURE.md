# 项目架构与目录说明（逐文件展开）

本项目实现基于四面体的 Radiance Meshes：加载 COLMAP/Blender 数据 → 点云或随机点云 → Delaunay 四面体化 → iNGP 哈希编码输出密度/颜色/梯度/SH → 光线积分渲染 → densify/cull → 冻结或网格导出。下面按文件说明作用与主要逻辑。

## 根目录脚本与配置
- `train.py`：主训练循环。解析 `utils/args.Args`；`data/loader.load_dataset` 读取相机/点云并 PCA 对齐；`Model.init_from_pcd` 构建四面体；循环中按间隔重建 Delaunay、densify/cull、提升 SH 阶、在 `freeze_start` 调 `models.frozen.freeze_model`；写 ckpt、训练视频。
- `evaluate.py`：纯 Python 渲染评估，加载 ckpt，遍历相机输出 PSNR/图像。
- `evaluate_traced.py`：使用 Torch traced 模型评估以减少 Python overhead。
- `evaluate_traced_optix.py`：调用 tracers（OptiX/CUDA）进行加速评估。
- `render_fisheye.py`：鱼眼投影渲染入口。
- `tile_benchmark.py`：瓦片化渲染性能基准。
- `convert.py`：把自备数据/图片转换到期望的目录结构、可选缩放。
- `download_n_benchmark*.sh`：下载数据并跑基准；`run*.sh`：示例批处理入口。
- `hyperparam.py`：读取 CSV 队列（`hyperparams/`）批量生成训练命令，调度 GPU，写结果 CSV。
- 资源/配置：`pyproject.toml`（依赖与 uv 配置）、`README.md`（使用示例）、`MANIFEST.in`、`LICENSE`、`uv.lock`。

## 数据层（data/）
- `camera.py`：`Camera` 类封装投影矩阵、世界/相机变换、畸变参数、图像张量；提供更新和缓存。
- `colmap_loader.py`：解析 COLMAP cameras/images/points3D 的二进制与文本版本；`qvec2rotmat` 处理旋转四元数；返回结构体供 dataset 读取。
- `dataset_readers.py`：数据集入口回调字典。`readColmapSceneInfo` 读取 COLMAP 相机、生成/缓存 `points3D.ply`、包装为 `SceneInfo`；`readNerfSyntheticInfo` 读取 Blender transforms_train/test，必要时生成随机点云；`storePly`/`fetchPly` 辅助；`getNerfppNorm` 计算归一化参数。
- `loader.py`：高层加载。`load_cam` 读取图像按分辨率缩放，构造 `Camera`；`transform_cameras_pca` 计算 PCA 对齐并作用于相机与点云；`load_dataset` 选择 Colmap/Blender 读取、调用加载相机、变换点云并返回训练/测试相机与 `SceneInfo`。
- `types.py`：`BasicPointCloud`、`CameraInfo`、`SceneInfo`、`ProjectionType` 类型定义。

## 模型与优化（models/）
- `ingp_color.py`：
  - `iNGPDW`：iNGP 哈希编码（PyTorch 或 tinycudann），多头 MLP 输出密度/颜色/梯度/SH；支持固定/随机球面采样，截断高斯半径。
  - `Model`：核心四面体场景。`init_from_pcd` 下采样点云、加小噪声、生成球面外壳点（Fibonacci），用 `gDel3D.Del` 做 3D Delaunay，记录中心/尺度；`update_triangulation` 重建 Delaunay、修正体积符号、可按密度/alpha 过滤；`compute_batch_features` 在四面体上算 circumcenter/radius → 哈希编码 → 密度/颜色/梯度/SH；`get_cell_values` 聚合特征用于渲染；`compute_features`、`calc_tet_density` 等做分析/导出。
  - `TetOptimizer`：三路 Adam（编码/网络/顶点），指数与 Spiking LR 调度；管理 densify/cull/freezing 时机；缓存顶点梯度。
- `base_model.py`：基础接口/属性，供不同模型继承。
- `frozen.py`：冻结版本的模型/优化器，压缩参数量，支持后续微调或导出。

## 几何、渲染与训练工具（utils/）
- `args.py`：参数定义/解析，默认训练配置，JSON 读写。
- `camera_path.py`：生成相机飞行路径（z 轴采样、插值）。
- `cam_util.py`：相机采样/路径生成、焦距与高度调节、随机扰动。
- `compare_quad.py` / `compare_quad_rt.py`：对比四面体积分/渲染实现的测试脚本。
- `contraction.py`：空间压缩/反压缩（Instant-NGP 风格）。
- `densification.py`：统计贡献度、克隆/分裂/裁剪顶点，决定 densify/cull mask。
- `eval_sh_py.py`：纯 Python 球谐评估。
- `gradient.py`：梯度相关辅助函数。
- `graphics_utils.py`：矩阵/向量、光线构造、归一化等基础图形学操作。
- `hashgrid.py`：PyTorch HashGrid/iNGP 实现，含偏移与参数存取。
- `ingp_util.py`：hashgrid 参数计算、偏移。
- `jax_math.py` / `safe_math.py`：梯度安全的数学运算（clip、log/exp、幂等）。
- `lib_bilagrid.py`：双边网格数据结构、总变差正则、切片。
- `loss_utils.py`：常用损失（L1/SSIM 混合等）。
- `mesh_util.py`：从四面体提取/合并三角网格、面邻接、导出 PLY、按体积/面数筛选。
- `model_util.py`：密度/颜色/梯度/SH 激活组合，SH 评价，顶点颜色估计，输出打包。
- `optim.py`：自定义 Adam（可忽略部分参数、控制 weight decay）。
- `ssim.py`：SSIM 计算（非 fused）。
- `stepfun.py`：分段函数/积分工具。
- `test_util.py`：小型性能/正确性测试助手。
- `topo_utils.py`：四面体体积、重心、邻接矩阵、Fibonacci 球点、扩张凸包等几何工具。
- `train_util.py`：学习率调度器（指数、Spiking）、采样器等。

## 光栅化与查表相关
- `delaunay_rasterization/__init__.py`：导出内部接口。
- `delaunay_rasterization/internal/render_err.py` / `render_grid.py`：Slang/Torch 管线，对四面体做光栅化、误差计算。
- `alphablend_tiled_slang.py` / `alphablend_tiled_slang_interp.py` / `tile_shader_slang.py`：瓦片式 alpha blend 着色器 Python 包装。
- `sort_by_keys`：CUDA 排序辅助。
- `util.py`：内部工具。
- `slang/`：Slang 着色器与模块（tile shader、camera、safe-math、intersect 等）。
- `dtlookup/dtlookup.py` / `dtwalk.py`：距离场查表/走访；`dtlookup/slang/` 保存相关 Slang 内核。
- `jaxutil/tetra_quad.py` / `jaxutil/safe_math.py`：JAX 版本的四面体积分与安全数学。

## 低层追踪与 CUDA（tracers/）
- 构建支持：根 `CMakeLists.txt`，`cmake/` 与 `third_party/` 提供 FindCUDA/FindOptiX；`utils/`、`tests/`、`debug_tools/` 提供调试/单测脚本（如畸变损失、迭代剥离）。
- `ever/fast_ellipsoid_splinetracer.py`：椭球追踪 Python 包装。子目录：
  - `create_aabbs.*`：生成 AABB。
  - `initialize_density.*`：初始化密度缓冲（CUDA）。
  - `Forward.*` / `splinetracer.cpp` / `shaders.cu`：椭球前向追踪核心。
  - `py_binding.cpp`：pybind11 绑定；`scene.h`、`structs.h`、`GAS.*`、`CUDABuffer.h`、`cuda_util.h`：场景和缓冲封装。
  - `slang/`：对应 Slang 着色器。
- `splinetracers/tetra_splinetracer.py`：四面体追踪 Python 包装。`tetra_splinetracer/` 目录：
  - `Forward.cpp` / `Backward.cpp`：前/反向 CUDA 追踪。
  - `optixTriangle.*`、`TriangleMesh.*`：OptiX 三角形路径。
  - `py_binding.cpp`：pybind11 绑定；共享 `scene.h`、`structs.h`、`GAS.*`、`CUDABuffer.h`、`cuda_util.h`。
  - `slang/` 与 `splinetracers/slang/`：OptiX/Slang 内核、SH 计算、张量工具。
- `splinetracers/quad.py`、`ever/quad.py`：轻量包装/测试四面体或椭球追踪。

## 脚本与批量实验（scripts/、hyperparam*、hyperparams/）
- `scripts/extract_mesh.py`：从 ckpt 计算贡献度/SH/梯度，提取网格并写 PLY。
- `render_smooth.py` / `render_smooth_pulsing.py` / `render_smooth_iter.py`：平滑/脉冲/迭代方式渲染训练过程。
- `render_gradient.py`：渲染梯度可视化。
- `export_spinning.py`：导出旋转视角序列；`collect_stats.bash` / `process_stats.py` 收集与处理实验统计。
- `run_full*.csv`、`run_m360*.csv` 等：批量实验清单；`train_smalls.bash` 等脚本作为模板运行。
- `hyperparam.py`：读取 CSV 生成命令，调度 GPU，收集结果。
- `hyperparams/`：网格搜索/实验的输入输出 CSV。

## 其他
- `tests/`：针对 Delaunay 更新、相机加载、单/多四面体渲染、内存泄漏等快速测试脚本。
- `gDel3D/`：Delaunay 四面体化子模块（C++/pybind，供 `Model.update_triangulation`）。
- `submodules/fused-ssim/`：fused SSIM CUDA 实现。
- `assets/`、`notebooks/`、`results/`、`output/`：资源、调研 Notebook、默认输出目录。
