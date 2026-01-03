# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
# Copyright (c) 2017 Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun
# ----------------------------------------------------------------------------
"""
TNT (Tanks and Temples) evaluation configuration.

tau 是 F-score 计算的距离阈值（单位：米），不同场景有不同的阈值。
"""

# 各场景的 tau 值（距离阈值，单位：米）
SCENES_TAU_DICT = {
    # Intermediate scenes
    "Barn": 0.01,
    "Caterpillar": 0.005,
    "Church": 0.025,
    "Courthouse": 0.025,
    "Ignatius": 0.003,
    "Meetingroom": 0.01,
    "Truck": 0.005,
    # Advanced scenes
    "Auditorium": 0.01,
    "Ballroom": 0.01,
    "Courtroom": 0.01,
    "Museum": 0.01,
    "Palace": 0.01,
    "Temple": 0.01,
}

# TNT 场景分组
INTERMEDIATE_SCENES = ["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
ADVANCED_SCENES = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]


def get_scene_tau(scene_name: str) -> float:
    """获取场景的 tau 值"""
    if scene_name not in SCENES_TAU_DICT:
        raise ValueError(f"Unknown scene: {scene_name}. Available: {list(SCENES_TAU_DICT.keys())}")
    return SCENES_TAU_DICT[scene_name]


def get_scene_category(scene_name: str) -> str:
    """获取场景类别（intermediate 或 advanced）"""
    if scene_name in INTERMEDIATE_SCENES:
        return "intermediate"
    elif scene_name in ADVANCED_SCENES:
        return "advanced"
    else:
        return "unknown"
