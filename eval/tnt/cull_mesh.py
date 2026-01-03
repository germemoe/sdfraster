"""
TNT mesh culling using camera visibility.

Reference: MILo/milo/eval/tnt/cull_mesh.py

与 DTU 的 cull_mesh 不同，TNT 使用深度渲染来判断顶点可见性。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from copy import deepcopy

# pyrender 需要设置 EGL 渲染
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# 延迟导入 pyrender（可能未安装）
_pyrender = None


def _get_pyrender():
    """延迟加载 pyrender 模块"""
    global _pyrender
    if _pyrender is None:
        try:
            import pyrender
            _pyrender = pyrender
        except ImportError:
            raise ImportError(
                "pyrender is required for TNT mesh culling. "
                "Install it with: pip install pyrender"
            )
    return _pyrender


def extract_depth_from_mesh(
    mesh: trimesh.Trimesh,
    c2w_list: list,
    H: int, W: int,
    fx: float, fy: float, cx: float, cy: float,
    far: float = 20.0,
) -> list:
    """
    从 mesh 渲染深度图。

    Args:
        mesh: Trimesh 对象
        c2w_list: 相机位姿列表（c2w，4x4 torch.Tensor）
        H, W: 图像尺寸
        fx, fy, cx, cy: 相机内参
        far: 远平面距离

    Returns:
        深度图列表（torch.Tensor）
    """
    pyrender = _get_pyrender()

    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)

    renderer = pyrender.OffscreenRenderer(W, H)
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.SKIP_CULL_FACES

    depths = []
    for c2w in c2w_list:
        if isinstance(c2w, torch.Tensor):
            c2w = c2w.detach().cpu().numpy()
        c2w_gl = deepcopy(c2w)
        scene.set_pose(camera_node, c2w_gl)
        depth = renderer.render(scene, flags)
        depths.append(torch.from_numpy(depth))

    renderer.delete()
    return depths


class MeshCuller:
    """
    Mesh 裁剪器：基于相机可见性裁剪 mesh。

    Reference: MILo/milo/eval/tnt/cull_mesh.py Mesher class
    """

    def __init__(
        self,
        H: int, W: int,
        fx: float, fy: float, cx: float, cy: float,
        far: float = 20.0,
        min_visible_views: int = 20,
        points_batch_size: int = 500000,
    ):
        """
        Args:
            H, W: 图像尺寸
            fx, fy, cx, cy: 相机内参
            far: 远平面距离
            min_visible_views: 最少可见视图数（低于此值的顶点被剔除）
            points_batch_size: 批处理大小
        """
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.far = far
        self.min_visible_views = min_visible_views
        self.points_batch_size = points_batch_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.eps = 0.005  # 深度容差

    @torch.no_grad()
    def compute_visibility_mask(
        self,
        vertices: np.ndarray,
        depth_list: list,
        c2w_list: list,
    ) -> np.ndarray:
        """
        计算顶点可见性 mask。

        Args:
            vertices: 顶点坐标 (N, 3)
            depth_list: 深度图列表
            c2w_list: 相机位姿列表

        Returns:
            visibility mask (N,) bool
        """
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.from_numpy(vertices)
        vertices = vertices.clone().detach().float()

        mask_list = []

        for batch_verts in torch.split(vertices, self.points_batch_size, dim=0):
            n_pts = batch_verts.shape[0]
            valid_count = torch.zeros(n_pts, device=self.device, dtype=torch.int)

            for i, (depth, c2w) in enumerate(zip(depth_list, c2w_list)):
                points = batch_verts.to(self.device).float()
                if isinstance(c2w, torch.Tensor):
                    c2w = c2w.to(self.device).float()
                else:
                    c2w = torch.from_numpy(c2w).to(self.device).float()
                depth = depth.to(self.device)

                # 计算 w2c
                w2c = torch.inverse(c2w).float()

                # 转换到相机坐标系
                ones = torch.ones(n_pts, 1, device=self.device)
                homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1)
                cam_cord_homo = w2c @ homo_points
                cam_cord = cam_cord_homo[:, :3, :]  # (N, 3, 1)

                # 投影到图像平面
                K = torch.eye(3, device=self.device)
                K[0, 0], K[0, 2] = self.fx, self.cx
                K[1, 1], K[1, 2] = self.fy, self.cy

                uv = K @ cam_cord.float()
                z = uv[:, -1:] + 1e-8
                uv = uv[:, :2] / z  # (N, 2, 1)
                u, v = uv[:, 0, 0], uv[:, 1, 0]
                z = z[:, 0, 0]

                # 检查是否在视锥内
                in_frustum = (u >= 0) & (u <= self.W - 1) & (v >= 0) & (v <= self.H - 1) & (z > 0)

                # 采样深度
                depth_2d = depth.reshape(1, 1, self.H, self.W)
                vgrid = uv.reshape(1, 1, -1, 2)
                vgrid[..., 0] = (vgrid[..., 0] / (self.W - 1) * 2.0 - 1.0)
                vgrid[..., 1] = (vgrid[..., 1] / (self.H - 1) * 2.0 - 1.0)

                depth_sample = F.grid_sample(
                    depth_2d, vgrid, padding_mode='border', align_corners=True
                ).reshape(-1)

                # 检查是否在表面前方
                is_visible = torch.where(
                    depth_sample > 0.0,
                    z < (depth_sample + self.eps),
                    torch.ones_like(z, dtype=torch.bool)
                )

                in_frustum = in_frustum & is_visible
                valid_count += in_frustum.int()

            # 至少需要 min_visible_views 个视图可见
            mask = valid_count >= self.min_visible_views
            mask_list.append(mask.cpu().numpy())

        return np.concatenate(mask_list, axis=0)

    def cull_mesh(
        self,
        mesh: trimesh.Trimesh,
        c2w_list: list,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """
        根据相机可见性裁剪 mesh。

        Args:
            mesh: 输入 mesh
            c2w_list: 相机位姿列表

        Returns:
            裁剪后的 mesh
        """
        if verbose:
            print(f"[TNT-CULL] Culling mesh with {len(c2w_list)} cameras...")
            print(f"    Original: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # 渲染深度图
        if verbose:
            print("    Rendering depth maps...")
        depth_list = extract_depth_from_mesh(
            mesh, c2w_list,
            H=self.H, W=self.W,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            far=self.far
        )

        # 计算可见性 mask
        if verbose:
            print("    Computing visibility mask...")
        vertices = mesh.vertices[:, :3]
        mask = self.compute_visibility_mask(vertices, depth_list, c2w_list)

        if verbose:
            print(f"    Visible vertices: {mask.sum()} / {len(mask)} ({100 * mask.mean():.1f}%)")

        # 裁剪 mesh
        face_mask = mask[mesh.faces].all(axis=1)
        culled_mesh = deepcopy(mesh)
        culled_mesh.update_faces(face_mask)
        culled_mesh.remove_unreferenced_vertices()

        if verbose:
            print(f"    After cull: {len(culled_mesh.vertices)} verts, {len(culled_mesh.faces)} faces")

        return culled_mesh


def cull_mesh_with_cameras(
    mesh: trimesh.Trimesh,
    train_cameras: list,
    min_visible_views: int = 20,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    使用训练相机裁剪 mesh（便捷函数）。

    Args:
        mesh: 输入 mesh
        train_cameras: 训练相机列表（需要有 c2w, fx, fy, cx, cy, H, W 属性）
        min_visible_views: 最少可见视图数
        verbose: 是否输出详细信息

    Returns:
        裁剪后的 mesh
    """
    if len(train_cameras) == 0:
        return mesh

    # 获取相机参数（假设所有相机有相同的内参和尺寸）
    cam0 = train_cameras[0]
    H = cam0.image_height
    W = cam0.image_width

    # 从 FOV 计算焦距
    import math
    fx = W / (2 * math.tan(cam0.fovx / 2))
    fy = H / (2 * math.tan(cam0.fovy / 2))
    cx = W / 2.0
    cy = H / 2.0

    # 提取 c2w 矩阵
    c2w_list = []
    for cam in train_cameras:
        c2w = (cam.world_view_transform.T).inverse()
        c2w_list.append(c2w)

    # 创建 culler 并执行裁剪
    culler = MeshCuller(
        H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
        min_visible_views=min_visible_views
    )

    return culler.cull_mesh(mesh, c2w_list, verbose=verbose)
