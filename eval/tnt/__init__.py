# TNT (Tanks and Temples) evaluation module
from .evaluate import evaluate_tnt_mesh, get_tnt_info, load_pca_transform, align_mesh_to_tnt
from .config import get_scene_tau, get_scene_category, SCENES_TAU_DICT, INTERMEDIATE_SCENES, ADVANCED_SCENES
from .cull_mesh import cull_mesh_with_cameras, MeshCuller
from .registration import (
    CameraPose,
    read_trajectory,
    trajectory_alignment,
    refine_registration,
)
from .eval_core import evaluate_point_clouds

__all__ = [
    # Main evaluation
    'evaluate_tnt_mesh',
    'get_tnt_info',
    'load_pca_transform',
    'align_mesh_to_tnt',
    # Config
    'get_scene_tau',
    'get_scene_category',
    'SCENES_TAU_DICT',
    'INTERMEDIATE_SCENES',
    'ADVANCED_SCENES',
    # Mesh culling
    'cull_mesh_with_cameras',
    'MeshCuller',
    # Registration
    'CameraPose',
    'read_trajectory',
    'trajectory_alignment',
    'refine_registration',
    # Core evaluation
    'evaluate_point_clouds',
]
