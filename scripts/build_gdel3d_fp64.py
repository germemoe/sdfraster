#!/usr/bin/env python3
"""Build the pinned gDel3D (float64 + final-alive tet export) in an external copy."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

SOURCE_REVISION = "105b21784e98ee28844833e92073dcf2c21d07a8"
PATCH_SHA256 = "d05a77818c6a1380a0ec854d18d1647f1930f370cd9bcf97d5262d1c56307a69"
ALIVE_PATCH_SHA256 = "7a6dbccf78579f4cb1a8bbcd4379d85b1bd8942288e62d02a4df1744c290568a"
SOURCE_FILES = {
    "GDelFlipping/src/gDel3D/CommonTypes.h": (
        "bcef6c2dc403674adebe8f7f6d1e72927a4290fd2e2967f1696c597ff727cecd",
        "e1f2c056cba7b829cb34a25d51461e9fb0c844035e3251dd3dd9aaa317dbe31a"),
    "GDelFlipping/src/bindings.cpp": (
        "06e886781f07d90c0ba13ef78623a8ad6fb12ad760d795837f0c1a2c821d3a16",
        "d0b31d1e5b5fbc7b54e714c4ea9947a302f13854c85c0a2ad1be0129a187ab7e"),
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_files(source, patched=False):
    for relative, hashes in SOURCE_FILES.items():
        if digest(source / relative) != hashes[int(patched)]:
            raise RuntimeError(f"Dependency source hash mismatch: {source / relative}")

def prepare_source(original, destination, patch, alive_patch):
    if digest(patch) != PATCH_SHA256:
        raise RuntimeError("FP64 dependency patch hash mismatch")
    if digest(alive_patch) != ALIVE_PATCH_SHA256:
        raise RuntimeError("Final-alive export patch hash mismatch")
    verify_files(original)
    if destination.exists():
        verify_files(destination, patched=True)
        return
    staging = Path(tempfile.mkdtemp(prefix="gdel3d-prepare-", dir=destination.parent))
    try:
        copied = staging / "source"
        shutil.copytree(original, copied, ignore=shutil.ignore_patterns(
            ".git", "build", "__pycache__", "*.pyc"))
        for item in (patch, alive_patch):
            subprocess.run(["patch", "--batch", "--forward", "-p1", "--input", str(item)],
                           cwd=copied, check=True)
        verify_files(copied, patched=True)
        copied.rename(destination)
    finally:
        shutil.rmtree(staging)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=repository / "third_party" / "gdel3d")
    parser.add_argument("--build-root", type=Path, required=True,
                        help="External build directory; original submodule is never edited")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-home", type=Path,
                        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    parser.add_argument("--cuda-arch", type=int, default=89)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--install", action="store_true",
                        help="Atomically copy the built module into the selected interpreter's platlib")
    parser.add_argument("--probe-points", type=int, default=0,
                        help="Optional bounded CUDA float64 Delaunay correctness probe (at least 8)")
    args = parser.parse_args()
    if args.jobs < 1 or args.cuda_arch < 1 or (args.probe_points and args.probe_points < 8):
        parser.error("jobs/arch must be positive; probe-points must be 0 or at least 8")
    args.build_root = args.build_root.resolve()
    args.source = args.source.resolve()
    args.build_root.mkdir(parents=True, exist_ok=True)
    copied = args.build_root / "source-fp64-alive"
    build = args.build_root / "build"
    patch = repository / "patches/gdel3d-fp64-realtype.patch"
    alive_patch = repository / "patches/gdel3d-final-alive-export.patch"
    status_path = args.build_root / "manifest.json"
    log_path = args.build_root / "build.log"
    state = {"status": "running", "source_revision": SOURCE_REVISION,
             "original_source": str(args.source), "patched_source": str(copied),
             "patch": str(patch), "patch_sha256": PATCH_SHA256,
             "python": args.python, "cuda_arch": args.cuda_arch, "jobs": args.jobs,
             "precision": "float64", "steps": [], "log": str(log_path)}
    def save():
        state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_path.write_text(json.dumps(state, indent=2))
    env = os.environ.copy()
    env.update(CUDA_HOME=str(args.cuda_home), CUDACXX=str(args.cuda_home / "bin/nvcc"),
               MAX_JOBS=str(args.jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
    env["PATH"] = str(Path(args.python).absolute().parent) + ":" + str(args.cuda_home / "bin") + ":" + env.get("PATH", "")
    def run(name, command):
        entry = {"name": name, "command": command, "status": "running"}
        state["steps"].append(entry); save()
        with log_path.open("a") as log:
            log.write("\nSTEP " + name + "\n"); log.flush()
            process = subprocess.run(command, env=env, cwd=args.build_root,
                                     stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=process.returncode,
                     status="passed" if process.returncode == 0 else "failed")
        save()
        if process.returncode:
            raise RuntimeError(f"{name} failed with exit {process.returncode}; see {log_path}")
    save()
    try:
        prepare_source(args.source, copied, patch, alive_patch)
        run("configure", ["cmake", "-S", str(copied), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
            "-DPython3_EXECUTABLE=" + args.python, "-DUSE_SINGLE_PRECISION=OFF"])
        run("compile", ["cmake", "--build", str(build), "--target", "gdel3d.so",
                        "--parallel", str(args.jobs)])
        module = build / "gdel3d.so"
        state["module"] = str(module); state["module_sha256"] = digest(module)
        # Force the build directory first so an existing installed module cannot
        # make the dtype or correctness probe pass on the wrong binary.
        probe = "import sys,json,torch; sys.path.insert(0," + repr(str(build)) + "); import gdel3d; assert gdel3d.get_dtype()=='float64'; result={'dtype':gdel3d.get_dtype(),'module':gdel3d.__file__}"
        if args.probe_points:
            probe += "; torch.manual_seed(52); points=torch.rand((" + str(args.probe_points) + ",3),dtype=torch.float64); delaunay=gdel3d.Del(len(points)); tetrahedra,output=delaunay.compute(points); valid=bool(output.check_correctness(points)); assert valid; result.update(input_dtype=str(points.dtype),points=len(points),tetrahedra=len(tetrahedra),check_correctness=valid)"
        probe += "; print(json.dumps(result))"
        run("actual_dtype_and_optional_cuda_probe", [args.python, "-c", probe])
        if args.install:
            platlib = subprocess.check_output([args.python, "-c", "import sysconfig; print(sysconfig.get_path('platlib'))"], env=env, text=True).strip()
            destination = Path(platlib) / "gdel3d.so"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".so.tmp")
            shutil.copy2(module, temporary); temporary.replace(destination)
            state["installed_module"] = str(destination)
        state["status"] = "completed"; save()
    except Exception:
        state.update(status="failed", error=traceback.format_exc()); save(); raise
    print(json.dumps({"status": state["status"], "manifest": str(status_path),
                      "module": str(module), "dtype": "float64"}))

if __name__ == "__main__":
    main()
