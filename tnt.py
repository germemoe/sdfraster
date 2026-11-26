#!/usr/bin/env python3
"""Train and evaluate the SDFRaster Tanks and Temples presets.

The preset only depends on the scene type: Meetingroom is indoor, the other
five training scenes are outdoor. Evaluation extracts the raw Marching
Tetrahedra (MT) mesh of the final checkpoint and scores it with the legacy
Tanks and Temples protocol used by MILo.
"""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "outdoor": ROOT / "configs" / "tnt_outdoor.json",
    "indoor": ROOT / "configs" / "tnt_indoor.json",
}
SCENE_TYPES = {
    "Barn": "outdoor", "Caterpillar": "outdoor", "Courthouse": "outdoor",
    "Ignatius": "outdoor", "Truck": "outdoor", "Meetingroom": "indoor",
}


def scene_type(name, override):
    if override is not None:
        return override
    if name not in SCENE_TYPES:
        raise ValueError(f"Unknown scene {name!r}; pass --scene-type indoor|outdoor")
    return SCENE_TYPES[name]


def train(args):
    dataset = args.data.absolute()
    if not (dataset / "sparse" / "0").is_dir():
        raise FileNotFoundError(dataset / "sparse" / "0")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new training output directory: {out}")
    config = (args.config or CONFIGS[scene_type(dataset.name, args.scene_type)]).resolve()
    cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(config),
           "--dataset_path", str(dataset), "--output_path", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def evaluate(args):
    checkpoint = args.checkpoint.resolve()
    for name in ("ckpt.pth", "config.json", "poses.npy", "transform.txt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    gt = args.gt.resolve()
    out = args.out.resolve() if args.out else checkpoint / "eval_tnt"
    if out.exists():
        raise FileExistsError(f"Choose a new evaluation output directory: {out}")
    out.mkdir(parents=True)

    import numpy as np
    import open3d as o3d
    import torch
    from models.ingp_color import Model
    from utils.matmul_precision import (checkpoint_float32_matmul_precision,
                                        configure_float32_matmul_precision)
    from utils.tetmesh import marching_tetrahedra

    if not torch.cuda.is_available():
        raise RuntimeError("MT extraction requires CUDA")
    config = json.loads((checkpoint / "config.json").read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configure_float32_matmul_precision(checkpoint_float32_matmul_precision(checkpoint))
    model = Model.load_ckpt(checkpoint, torch.device("cuda"))
    model.eval()
    with torch.no_grad():
        sdf = model.query_vert_sdf()
        vertices, faces = marching_tetrahedra(model.vertices, model.indices, sdf,
                                              num_internal=model.num_int_verts)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    if len(mesh.triangles) == 0:
        raise RuntimeError("Marching tetrahedra returned an empty surface")
    mesh_path = out / "mesh_mt.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Could not save {mesh_path}")
    del model, sdf, vertices, faces, mesh
    torch.cuda.empty_cache()

    # The legacy evaluator restores the mesh to the COLMAP frame with the saved
    # PCA transform and registers the training trajectory to the official log.
    sys.path.insert(0, str(ROOT / "eval" / "tnt_milo"))
    import paper_legacy
    result = paper_legacy.evaluate(mesh_path, checkpoint / "poses.npy",
                                   checkpoint / "transform.txt", gt, out / "score")
    names = ("precision", "recall", "fscore")
    if not all(math.isfinite(result[name]) for name in names):
        raise RuntimeError("TNT evaluation did not produce finite metrics")
    metrics = {name: float(result[name]) for name in names}
    metrics.update(scene=result["scene"], threshold=result["threshold"], mesh="mesh_mt.ply",
                   protocol=result["protocol"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"TNT {metrics['scene']} F1: {metrics['fscore']:.6f} "
          f"(P {metrics['precision']:.4f}, R {metrics['recall']:.4f})")
    print(f"Saved results to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", help="Train the TNT preset to step 18,000")
    training.add_argument("--data", type=Path, required=True,
                          help="MILo COLMAP scene directory, e.g. /data/TNT/Barn")
    training.add_argument("--out", type=Path, required=True, help="New run directory")
    training.add_argument("--scene-type", choices=sorted(CONFIGS),
                          help="Default: inferred from the TNT scene name")
    training.add_argument("--config", type=Path, help="Override the preset JSON")

    scoring = commands.add_parser("eval", help="Export the raw MT mesh and score it")
    scoring.add_argument("--checkpoint", type=Path, required=True, help="Training output directory")
    scoring.add_argument("--gt", type=Path, required=True,
                         help="Official TNT GT directory of this scene, e.g. /data/GT_TNT/Barn")
    scoring.add_argument("--out", type=Path, help="Default: <checkpoint>/eval_tnt")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
