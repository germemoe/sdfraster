  # Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import slangtorch
import os

shaders_path = os.path.dirname(__file__)
# SlangTorch writes metadata beside sources. CPFS imports from independent
# container hosts must not share this writable cache namespace.
_shader_cache_root = os.environ.get("SDFRASTER_SHADER_CACHE_ROOT")
if _shader_cache_root:
    import hashlib
    import socket
    from pathlib import Path
    import shutil
    original = Path(shaders_path)
    shader_files = sorted(p for p in original.rglob("*")
                          if p.is_file() and ".slangtorch_cache" not in p.parts
                          and p.suffix in (".slang", ".slangh", ".h"))
    digest = hashlib.sha256()
    for file in shader_files:
        digest.update(str(file.relative_to(original)).encode())
        digest.update(file.read_bytes())
    namespace = Path(_shader_cache_root) / socket.gethostname() / digest.hexdigest()[:16]
    namespace.mkdir(parents=True, exist_ok=True)
    for file in shader_files:
        target = namespace / file.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != file.read_bytes():
            # A same-host concurrent import should never observe a partial copy.
            temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
            shutil.copy2(file, temporary)
            temporary.replace(target)
    shaders_path = str(namespace)

TILE_SIZES_HW = [(4,4)]#, (8,8), (16,16)]

vertex_shader = slangtorch.loadModule(os.path.join(shaders_path, "vertex_shader.slang"))
tile_shader = slangtorch.loadModule(os.path.join(shaders_path, "tile_shader.slang"))
alpha_blend_shaders = {}
for tile_height, tile_width in TILE_SIZES_HW:
  alpha_blend_shaders[(tile_height, tile_width)] = slangtorch.loadModule(os.path.join(shaders_path, "alphablend_shader.slang"), 
                                                                         defines={"PYTHON_TILE_HEIGHT": tile_height, "PYTHON_TILE_WIDTH": tile_width})

alpha_blend_shaders_interp = {}
for tile_height, tile_width in TILE_SIZES_HW:
  alpha_blend_shaders_interp[(tile_height, tile_width)] = slangtorch.loadModule(os.path.join(shaders_path, "alphablend_shader_interp.slang"), 
                                                                         defines={"PYTHON_TILE_HEIGHT": tile_height, "PYTHON_TILE_WIDTH": tile_width})

_alpha_blend_interp_variants = {}


def get_alpha_blend_shader_interp(tile_height, tile_width, tet_normal_mode="raw"):
    if tet_normal_mode not in ("raw", "unit"):
        raise ValueError("tet_normal_mode must be raw or unit")
    tile = (tile_height, tile_width)
    if tile not in alpha_blend_shaders_interp:
        raise ValueError(f"Unsupported tile size {tile}")
    if tet_normal_mode == "raw":
        return alpha_blend_shaders_interp[tile]
    key = (tile_height, tile_width, tet_normal_mode)
    if key not in _alpha_blend_interp_variants:
        _alpha_blend_interp_variants[key] = slangtorch.loadModule(
            os.path.join(shaders_path, "alphablend_shader_interp.slang"),
            defines={"PYTHON_TILE_HEIGHT": tile_height,
                     "PYTHON_TILE_WIDTH": tile_width,
                     "SDFRASTER_TET_NORMAL_UNIT": 1})
    return _alpha_blend_interp_variants[key]
