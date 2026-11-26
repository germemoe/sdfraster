"""Evaluate a native MT mesh with the pinned MILo/Tanks and Temples legacy CLI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np


PROTOCOL = "tnt_legacy_milo_0c38aa6"
UPSTREAM = Path(__file__).with_name("legacy_milo")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _count(path):
    with path.open("rb") as stream:
        for row in stream:
            if row.startswith(b"element vertex "):
                return int(row.split()[2])
            if row.strip() == b"end_header":
                break
    raise ValueError(f"No vertex element in {path}")


def _restore_pose_frame(source, target, pca):
    import trimesh

    mesh = trimesh.load_mesh(source, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("TNT scoring requires a nonempty triangular mesh.")
    inv = np.linalg.inv(pca)
    vertices = np.asarray(mesh.vertices) @ inv[:3, :3].T + inv[:3, 3]
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite positions in the pose-frame mesh.")

    # Open3D writes Float64 PLY coordinates and preserves face order on export.
    import open3d as o3d
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                      o3d.utility.Vector3iVector(mesh.faces))
    if not o3d.io.write_triangle_mesh(str(target), result):
        raise OSError(f"Could not write {target}")


def evaluate(mesh, poses, transform, gt, output):
    """Use pre-PCA camera poses and the inverse-PCA mesh, then run legacy."""
    started = time.perf_counter()
    mesh, poses, transform, gt, output = map(lambda p: Path(p).resolve(),
                                              (mesh, poses, transform, gt, output))
    scene = gt.name
    for file in (mesh, poses, transform, gt / f"{scene}.ply", gt / f"{scene}.json",
                 gt / f"{scene}_COLMAP_SfM.log", gt / f"{scene}_trans.txt"):
        if not file.is_file():
            raise FileNotFoundError(file)
    pca = np.loadtxt(transform)
    c2w = np.load(poses, allow_pickle=False)
    if pca.shape != (4, 4) or not np.isfinite(pca).all():
        raise ValueError("Expected a finite 4x4 transform.txt.")
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("Expected finite pre-PCA (N,4,4) camera poses.")
    with (gt / f"{scene}_COLMAP_SfM.log").open() as stream:
        lines = sum(bool(line.strip()) and not line.startswith("#") for line in stream)
    if lines != 5 * len(c2w):
        raise ValueError(f"{scene}: saved camera poses must match the official trajectory in number and order.")

    source = json.loads((UPSTREAM / "SOURCE.json").read_text())
    for name, entry in source.items():
        if _sha(UPSTREAM / name) != entry["sha256"]:
            raise ValueError(f"Legacy evaluator source digest mismatch: {name}")

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="tnt-legacy-", dir=output) as scratch:
        staged = Path(scratch) / "mesh_in_pose_frame.ply"
        _restore_pose_frame(mesh, staged, pca)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--upstream",
                   "--dataset-dir", str(gt), "--traj-path", str(poses),
                   "--ply-path", str(staged), "--out-dir", str(output), "--view-crop", "0"]
        subprocess.run(command, check=True, env=dict(os.environ, MPLBACKEND="Agg"))
    precision, recall, fscore, tau, _ = map(float, np.loadtxt(output / f"{scene}.prf_tau_plotstr.txt"))
    if not np.isfinite([precision, recall, fscore, tau]).all():
        raise ValueError("Legacy evaluator returned non-finite metrics.")
    results = dict(scene=scene, protocol=PROTOCOL, precision=precision, recall=recall, fscore=fscore,
                   threshold=tau, voxel_size=tau / 2,
                   num_source_points=_count(output / f"{scene}.precision.ply"),
                   num_target_points=_count(output / f"{scene}.recall.ply"),
                   mesh_sha256=_sha(mesh), poses_sha256=_sha(poses), transform_sha256=_sha(transform),
                   wall_seconds=time.perf_counter() - started)
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def _run_upstream(args):
    import open3d as o3d

    sys.path.insert(0, str(UPSTREAM))
    import evaluation

    original = evaluation.get_f1_score_histo2

    def histogram(threshold, filename, stretch, distances1, distances2, verbose=True):
        try:
            return original(threshold, filename, stretch, distances1, distances2, verbose)
        except ZeroDivisionError:
            bins = np.arange(0, threshold * stretch, threshold / 100)
            first, edges = np.histogram(distances1, bins)
            second, _ = np.histogram(distances2, bins)
            return [0., 0., 0., edges, first.cumsum() / len(distances1),
                    edges, second.cumsum() / len(distances2)]

    evaluation.get_f1_score_histo2 = histogram
    np.random.seed(52)
    o3d.utility.random.seed(52)
    sys.argv = [str(UPSTREAM / "run.py"), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--upstream"]:
        _run_upstream(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for arg in ("mesh", "poses", "transform", "gt", "output"):
            parser.add_argument(f"--{arg}", type=Path, required=True)
        print(json.dumps(evaluate(**vars(parser.parse_args()))))
