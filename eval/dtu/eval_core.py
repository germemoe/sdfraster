# adapted from https://github.com/jzhangbs/DTUeval-python
# and MILo: https://github.com/autonomousvision/milo
"""
DTU Mesh Evaluation Core

This module computes Chamfer distance between reconstructed mesh and ground truth point cloud.
Requires DTU dataset with ObsMask and Points/stl directories.
"""

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import json
import os


def sample_single_tri(input_):
    """Sample points from a single triangle."""
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    """Write visualization point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def _nn_jobs_for_workers(cpu_workers):
    """None preserves the legacy automatic Pool/all-core NN behavior."""
    if cpu_workers is not None and (isinstance(cpu_workers, bool) or
            not isinstance(cpu_workers, int) or cpu_workers < 1):
        raise ValueError("cpu_workers must be a positive integer or None")
    return -1 if cpu_workers is None else cpu_workers


def greedy_radius_downsample_mask(points, radius, method="batch", cpu_workers=None):
    """Preserve shuffled greedy selection; skip queries for already-removed rows."""
    nn_jobs = _nn_jobs_for_workers(cpu_workers)
    if method not in ("batch", "streaming"):
        raise ValueError(f"Unknown downsample method: {method}")
    mask = np.ones(points.shape[0], dtype=np.bool_)
    if method == "batch":
        engine = skln.NearestNeighbors(n_neighbors=1, radius=radius,
                                      algorithm="kd_tree", n_jobs=nn_jobs)
        engine.fit(points)
        neighbors = engine.radius_neighbors(points, radius=radius, return_distance=False)
        for curr, idxs in enumerate(neighbors):
            if not mask[curr]:
                continue
            mask[idxs] = False
            mask[curr] = True
    else:
        tree = skln.KDTree(points)
        for curr in range(len(points)):
            if not mask[curr]:
                continue
            idxs = tree.query_radius(points[curr:curr+1], r=radius,
                                     return_distance=False)[0]
            mask[idxs] = False
            mask[curr] = True
    return mask


def evaluate_mesh(
    mesh_path: str,
    scan_id: int,
    dataset_dir: str,
    vis_out_dir: str = None,
    downsample_density: float = 0.2,
    patch_size: float = 60,
    max_dist: float = 20,
    visualize_threshold: float = 10,
    seed: int = None,
    downsample_method: str = "batch",
    cpu_workers: int = None,
):
    """
    Evaluate mesh against DTU ground truth.

    Args:
        mesh_path: Path to input mesh file (PLY format, in DTU world coordinates)
        scan_id: DTU scan ID (e.g., 37)
        dataset_dir: Path to DTU dataset root (containing ObsMask/ and Points/)
        vis_out_dir: Output directory for visualization (optional)
        downsample_density: Sampling density threshold
        patch_size: Bounding box patch size
        max_dist: Maximum distance for evaluation
        visualize_threshold: Threshold for visualization
        cpu_workers: Positive sampling/NN worker limit; None preserves legacy automatic workers

    Returns:
        dict with 'mean_d2s', 'mean_s2d', 'overall' metrics
    """
    nn_jobs = _nn_jobs_for_workers(cpu_workers)
    mp.freeze_support()

    thresh = downsample_density

    pbar = tqdm(total=9)
    pbar.set_description('read data mesh')
    data_mesh = o3d.io.read_triangle_mesh(mesh_path)

    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)

    if len(triangles) == 0:
        print(f"[EVAL] Warning: mesh has no triangles")
        return {'mean_d2s': float('inf'), 'mean_s2d': float('inf'), 'overall': float('inf')}

    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool(processes=cpu_workers) as mp_pool:
        new_pts = mp_pool.map(
            sample_single_tri,
            ((n1[i, 0], n2[i, 0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1, 0]) for i in range(len(n1))),
            chunksize=1024
        )

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng(seed)
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=nn_jobs)
    mask = greedy_radius_downsample_mask(data_pcd, thresh, downsample_method, cpu_workers=cpu_workers)
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{dataset_dir}/ObsMask/ObsMask{scan_id}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = patch_size
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{dataset_dir}/Points/stl/stl{scan_id:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{dataset_dir}/ObsMask/Plane{scan_id}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    # Visualization
    if vis_out_dir is not None:
        os.makedirs(vis_out_dir, exist_ok=True)
        pbar.update(1)
        pbar.set_description('visualize error')
        vis_dist = visualize_threshold
        R = np.array([[1, 0, 0]], dtype=np.float64)
        G = np.array([[0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 0, 1]], dtype=np.float64)
        W = np.array([[1, 1, 1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color[np.where(inbound)[0][grid_inbound][in_obs]] = R * data_alpha + W * (1 - data_alpha)
        data_color[np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:, 0] >= max_dist]] = G
        write_vis_pcd(f'{vis_out_dir}/vis_{scan_id:03}_d2s.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
        stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G
        write_vis_pcd(f'{vis_out_dir}/vis_{scan_id:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()

    over_all = (mean_d2s + mean_s2d) / 2
    print(f"[EVAL] mean_d2s: {mean_d2s:.4f}, mean_s2d: {mean_s2d:.4f}, overall: {over_all:.4f}")

    results = {
        'mean_d2s': float(mean_d2s),
        'mean_s2d': float(mean_s2d),
        'overall': float(over_all),
        'downsample_seed': seed,
        'downsample_method': downsample_method,
        'cpu_workers': cpu_workers,
        'nn_jobs': nn_jobs,
    }

    if vis_out_dir is not None:
        with open(f'{vis_out_dir}/results.json', 'w') as fp:
            json.dump(results, fp, indent=2)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to input mesh')
    parser.add_argument('--scan', type=int, required=True, help='DTU scan ID')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to DTU dataset')
    parser.add_argument('--vis_out_dir', type=str, default=None, help='Output directory for visualization')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    parser.add_argument('--seed', type=int, default=52)
    parser.add_argument('--downsample-method', choices=('batch', 'streaming'), default='batch')
    parser.add_argument('--cpu-workers', type=int, default=14,
                        help='Positive worker limit for mesh sampling and NN queries (default: 14)')
    args = parser.parse_args()

    evaluate_mesh(
        mesh_path=args.data,
        scan_id=args.scan,
        dataset_dir=args.dataset_dir,
        vis_out_dir=args.vis_out_dir,
        downsample_density=args.downsample_density,
        patch_size=args.patch_size,
        max_dist=args.max_dist,
        visualize_threshold=args.visualize_threshold,
        seed=args.seed,
        downsample_method=args.downsample_method,
        cpu_workers=args.cpu_workers,
    )
