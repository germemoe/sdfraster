#!/usr/bin/env python3
"""Train and evaluate the SDFRaster DTU preset."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "dtu_18k.json"
SCENE_RATIOS = ROOT / "configs" / "dtu_sdf_densify_ratios.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scene_id(path):
    match = re.fullmatch(r"scan(\d+)", path.name)
    if match is None:
        raise ValueError(f"Expected a DTU scene directory named scan<ID>, got {path}")
    return int(match[1])


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def train(args):
    scene_id(args.data)
    dataset = args.data.absolute()
    if not (dataset / "sparse" / "0").is_dir():
        raise FileNotFoundError(dataset / "sparse" / "0")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new training output directory: {out}")

    config = args.config.resolve()
    cmd = [
        sys.executable, str(ROOT / "train.py"), "--config", str(config),
        "--dataset_path", str(dataset), "--output_path", str(out),
    ]
    if config == DEFAULT_CONFIG:
        ratio = read_json(SCENE_RATIOS).get(dataset.name)
        if ratio is not None:
            cmd.extend(["--sdf_densify_ratio", str(ratio)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_mesh(o3d, path, mesh):
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh extraction returned no surface: {path.name}")
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise RuntimeError(f"Could not save mesh: {path}")


def evaluate(args):
    checkpoint = args.checkpoint.resolve()
    config = read_json(checkpoint / "config.json")
    dataset = args.data.absolute() if args.data else Path(config["dataset_path"])
    scan = scene_id(dataset)
    gt = args.gt.resolve()
    for path in (
        checkpoint / "ckpt.pth",
        checkpoint / "transform.txt",
        dataset / "cameras.npz",
        gt / "ObsMask" / f"ObsMask{scan}_10.mat",
        gt / "ObsMask" / f"Plane{scan}.mat",
        gt / "Points" / "stl" / f"stl{scan:03d}_total.ply",
    ):
        require_file(path)

    out = args.out.resolve() if args.out else checkpoint / "eval"
    if out.exists():
        raise FileExistsError(f"Choose a new evaluation output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import open3d as o3d
    import torch
    from data import loader
    from eval.dtu.evaluate import evaluate_dtu_mesh
    from models.ingp_color import Model
    from scripts.extract_mesh_tsdf import TSDFExtractor, post_process_mesh
    from utils.matmul_precision import (checkpoint_float32_matmul_precision,
                                        configure_float32_matmul_precision)
    from utils.tetmesh import marching_tetrahedra

    if not torch.cuda.is_available():
        raise RuntimeError("DTU mesh extraction and scoring require CUDA")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    o3d.utility.random.seed(seed)
    torch.set_num_threads(1)
    configure_float32_matmul_precision(checkpoint_float32_matmul_precision(checkpoint))
    cameras, _, scene = loader.load_dataset(
        dataset, config["image_folder"], data_device=config["data_device"],
        eval=config["eval"], resolution=config["resolution"],
    )
    transform = np.loadtxt(checkpoint / "transform.txt")
    if not np.allclose(transform, scene.transform, rtol=0.0, atol=1e-5):
        raise RuntimeError("Loaded cameras do not match the checkpoint transform")

    model = Model.load_ckpt(checkpoint, torch.device("cuda"))
    model.eval()
    with TemporaryDirectory(prefix=".dtu-eval-", dir=out.parent) as tmp:
        work = Path(tmp) / "output"
        work.mkdir()
        with torch.no_grad():
            vertices, faces = marching_tetrahedra(
                model.vertices, model.indices, model.query_vert_sdf(),
                num_internal=model.num_int_verts,
            )
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            write_mesh(o3d, work / "mesh_mt.ply", mesh)
            del vertices, faces, mesh

            extractor = TSDFExtractor(
                model, depth_ratio=config["tsdf_depth_ratio"],
                tile_size=config["tile_size"], min_t=config["min_t"],
            )
            extractor.reconstruction(cameras)
            mesh = extractor.extract_mesh(
                voxel_size=config["tsdf_voxel_size"],
                sdf_trunc=config["tsdf_sdf_trunc"],
                depth_trunc=config["tsdf_depth_trunc"],
                mask_background=True,
            )
            mesh = post_process_mesh(mesh, cluster_to_keep=config["tsdf_num_cluster"])
            write_mesh(o3d, work / "mesh_tsdf.ply", mesh)
            del mesh, extractor, model
        torch.cuda.empty_cache()

        # The reference DTU evaluator creates an aligned mesh on disk for scoring.
        # Keep that temporary file outside the published evaluation outputs.
        metrics = evaluate_dtu_mesh(
            mesh_path=str(work / "mesh_tsdf.ply"), train_cameras=cameras,
            dtu_path=str(gt), scan_id=scan, output_dir=str(Path(tmp) / "score"),
            cull=True, dilation_radius=config["cull_dilation_radius"],
            dataset_path=str(dataset), model_output_path=str(checkpoint),
            seed=seed, downsample_method=args.downsample_method,
            cpu_workers=args.workers, write_visualizations=False,
        )
        names = ("overall", "mean_d2s", "mean_s2d")
        if not metrics or not all(math.isfinite(metrics[name]) for name in names):
            raise RuntimeError("DTU evaluation did not produce finite metrics")
        result = {name: float(metrics[name]) for name in names}
        result.update(scene=dataset.name, mesh="mesh_tsdf.ply", units="mm")
        (work / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError(out)
        work.rename(out)
    print(f"DTU {dataset.name} TSDF CD: {result['overall']:.6f} mm")
    print(f"Saved results to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train the DTU preset to step 18,000")
    training.add_argument("--data", type=Path, required=True, help="Prepared DTU scene directory")
    training.add_argument("--out", type=Path, required=True, help="New run directory")
    training.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Override the preset JSON")

    scoring = commands.add_parser("eval", help="Export MT and TSDF meshes and score TSDF")
    scoring.add_argument("--checkpoint", type=Path, required=True, help="Training output or checkpoint directory")
    scoring.add_argument("--gt", type=Path, required=True, help="Official DTU evaluation data")
    scoring.add_argument("--data", type=Path, help="Scene directory if the training data has moved")
    scoring.add_argument("--out", type=Path, help="Default: <checkpoint>/eval")
    scoring.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    scoring.add_argument("--downsample-method", choices=("batch", "streaming"), default="batch")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        if args.workers < 1:
            parser.error("--workers must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
