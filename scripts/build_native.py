#!/usr/bin/env python3
"""Build the optional exact duplicate-coordinate counter used by Delaunay validation.

Without this library utils/triangulation_validation.py falls back to a NumPy sort
that returns identical counts, only slower on multi-million point clouds.
"""
import argparse
import ctypes
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "exact_triplet_counts.cpp"
TARGET = ROOT / "utils" / "exact_triplet_counts_linux_x86_64.so"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++-11")
    args = parser.parse_args()
    # Keep IEEE semantics: the counter compares raw float64 bit patterns.
    command = [args.cxx, "-std=c++17", "-O3", "-fPIC", "-shared", "-fno-fast-math",
               "-ffp-contract=off", "-fno-associative-math", str(SOURCE), "-o", str(TARGET)]
    subprocess.run(command, check=True)
    library = ctypes.CDLL(str(TARGET))
    library.exact_triplet_counts_abi.restype = ctypes.c_char_p
    abi = library.exact_triplet_counts_abi()
    if abi != b"E164_EXACT_NATIVE_F64_TRIPLET_COUNTS_V1":
        raise RuntimeError(f"Unexpected ABI {abi!r}")
    print(f"Built {TARGET}")


if __name__ == "__main__":
    main()
