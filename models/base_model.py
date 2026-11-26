import torch
from torch import nn
from typing import Optional
import tinyplypy
import numpy as np

from data.camera import Camera
from utils.model_util import activate_output, RGB2SH


class BaseModel(nn.Module):

    def get_cell_values(
        self,
        camera: Camera,
        mask: Optional[torch.Tensor] = None,
        all_circumcenters: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
    ):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        if self.chunk_size is None:
            # [DENSITY-RELATED] compute_batch_features 返回密度
            cc, normalized, density, rgb, grd, sh = self.compute_batch_features(
                vertices, indices, start, end, circumcenters=all_circumcenters
            )
            # [DENSITY-RELATED] activate_output 接收密度参数
            cell_output = activate_output(
                camera.camera_center.to(self.device),
                density, rgb, grd, sh, indices,
                cc, vertices,
                self.max_sh_deg, self.max_sh_deg,
            )
            return normalized, cell_output
        else:
            outputs = []
            normed_cc = []
            start = 0
            for start in range(0, indices.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, indices.shape[0])
                # [DENSITY-RELATED] compute_batch_features 返回密度
                circumcenters, density, rgb, grd, sh = self.compute_batch_features(
                    vertices, indices, start, end, circumcenters=all_circumcenters)
                # [DENSITY-RELATED] activate_output 接收密度参数
                dvrgbs = activate_output(camera.camera_center.to(self.device),
                                         density, rgb, grd, sh, indices[start:end],
                                         circumcenters,
                                         vertices, self.current_sh_deg, self.max_sh_deg)
                normed_cc.append(normalized)
                outputs.append(dvrgbs)
            features = torch.cat(outputs, dim=0)
            normed_cc = torch.cat(normed_cc, dim=0)
            return normed_cc, features

    def __len__(self):
        return self.vertices.shape[0]

    @property
    def num_int_verts(self):
        return self.contracted_vertices.shape[0]

    def get_circumcenters(self):
        circumcenter =  pre_calc_cell_values(
            self.vertices, self.indices, self.center, self.scene_scaling)
        return circumcenter

    def calc_tet_area(self):
        verts = self.vertices
        v0, v1, v2, v3 = verts[self.indices].unbind(dim=1)
        mat = torch.stack([v1-v0, v2-v0, v3-v0], dim=-1)
        return torch.det(mat).abs() / 6.0

    # [DENSITY-RELATED] 计算顶点密度 (从四面体密度聚合)
    def calc_vert_density(self):
        verts = self.vertices
        vertex_density = torch.zeros((verts.shape[0],), device=self.device)
        indices = self.indices.long()
        # [DENSITY-RELATED] 获取四面体密度
        density = self.calc_tet_density()
        reduce_type = "amax"
        # [DENSITY-RELATED] 将四面体密度散射到顶点
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 0], src=density, reduce=reduce_type)
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 1], src=density, reduce=reduce_type)
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 2], src=density, reduce=reduce_type)
        vertex_density.scatter_reduce_(dim=0, index=indices[..., 3], src=density, reduce=reduce_type)
        return vertex_density

    # [SDF-MODIFICATION] 基于 NeuS 公式计算四面体的最大可能 alpha
    def calc_tet_alpha(self, inv_s=None, indices=None, vert_sdf=None, **kwargs):
        """
        基于 NeuS 公式计算四面体的最大可能 alpha 贡献

        通过假设射线从 SDF 最大的顶点穿到最小的顶点，
        计算四面体可能贡献的 alpha 上界。

        Args:
            inv_s: NeuS 的 inv_s 参数，如果为 None 则从 backbone 获取
        Returns:
            alpha: (N,) 每个四面体的最大可能 alpha
        """
        if inv_s is None:
            inv_s = self.backbone.inv_s

        # 获取所有顶点的 SDF
        if vert_sdf is None:
            vert_sdf = self.query_vert_sdf()  # (M,)
        if indices is None:
            indices = self.indices

        # 获取每个四面体 4 个顶点的 SDF
        tet_vert_sdf = vert_sdf[indices.long()]  # (N, 4)

        sdf_max = tet_vert_sdf.max(dim=1)[0]  # (N,)
        sdf_min = tet_vert_sdf.min(dim=1)[0]  # (N,)

        # NeuS CDF: Φ(x) = sigmoid(x * inv_s)
        cdf_max = torch.sigmoid(sdf_max * inv_s)
        cdf_min = torch.sigmoid(sdf_min * inv_s)

        # 最大可能 alpha: (Φ(sdf_max) - Φ(sdf_min)) / Φ(sdf_max)
        alpha = ((cdf_max - cdf_min) / (cdf_max + 1e-10)).clamp(0, 1)
        return alpha

    # === [DENSITY-RELATED] 原密度 alpha 代码 (已注释) ===
    # def calc_tet_alpha(self, mode="min", density=None):
    #     alpha_list = []
    #     start = 0
    #     verts = self.vertices
    #     inds = self.indices
    #     v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
    #     edge_lengths = torch.stack([
    #         torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
    #         torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
    #     ], dim=0)
    #     if mode == "min":
    #         el = edge_lengths.min(dim=0)[0]
    #     elif mode == "max":
    #         el = edge_lengths.max(dim=0)[0]
    #     elif mode == "mean":
    #         el = edge_lengths.mean(dim=0)
    #     else:
    #         raise ValueError(f"Unknown mode: {mode}. Use 'min', 'max', or 'mean'.")
    #     density = self.calc_tet_density() if density is None else density
    #     alpha = 1 - torch.exp(-density.reshape(-1) * el.reshape(-1))
    #     return alpha
    # === [DENSITY-RELATED] 原密度代码结束 ===

    @torch.no_grad
    def save2ply(self, path):
        path.parent.mkdir(exist_ok=True, parents=True)

        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)  # shape (num_vertices, 3)

        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }

        N = self.indices.shape[0]
        sh_dim = ((self.max_sh_deg+1)**2-1)

        # [DENSITY-RELATED] compute_features 返回密度
        circumcenters, density, base_color_v0_raw, normed_grd, sh = self.compute_features(offset=True)

        base_color_v0_raw = base_color_v0_raw.cpu().numpy().astype(np.float32)
        grds = normed_grd.reshape(-1, 3).cpu().numpy().astype(np.float32)
        # [DENSITY-RELATED] 密度转换为numpy用于保存
        densities = density.reshape(-1).cpu().numpy().astype(np.float32)
        sh_coeffs = sh.reshape(-1, sh_dim, 3).cpu().numpy().astype(np.float32)

        tetra_dict = {}
        tetra_dict["indices"] = self.indices.cpu().numpy().astype(np.int32)
        # tetra_dict["mask"] = self.mask.cpu().numpy().astype(np.uint8)
        tetra_dict["s"] = np.ascontiguousarray(densities)  # [DENSITY-RELATED] 保存密度到ply
        for i, co in enumerate(["x", "y", "z"]):
            tetra_dict[f"grd_{co}"]         = np.ascontiguousarray(grds[:, i])

        sh_0 = RGB2SH(base_color_v0_raw)
        tetra_dict[f"sh_0_r"] = np.ascontiguousarray(sh_0[:, 0])
        tetra_dict[f"sh_0_g"] = np.ascontiguousarray(sh_0[:, 1])
        tetra_dict[f"sh_0_b"] = np.ascontiguousarray(sh_0[:, 2])
        for i in range(sh_coeffs.shape[1]):
            tetra_dict[f"sh_{i+1}_r"] = np.ascontiguousarray(sh_coeffs[:, i, 0])
            tetra_dict[f"sh_{i+1}_g"] = np.ascontiguousarray(sh_coeffs[:, i, 1])
            tetra_dict[f"sh_{i+1}_b"] = np.ascontiguousarray(sh_coeffs[:, i, 2])


        data_dict = {
            "vertex": vertex_dict,
            "tetrahedron": tetra_dict,
        }

        tinyplypy.write_ply(str(path), data_dict, is_binary=True)
