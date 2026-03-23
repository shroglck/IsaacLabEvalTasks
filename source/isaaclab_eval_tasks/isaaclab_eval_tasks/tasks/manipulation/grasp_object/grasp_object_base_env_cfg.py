# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as base_mdp
from isaaclab_eval_tasks.tasks.manipulation.mdp import is_object_lifted

from isaaclab_eval_tasks.tasks.manipulation.shared_base_env_cfg import ObjectTableSceneCfg, PickPlaceGR1T2EnvCfg, TerminationsCfg, EventCfg, ObservationsCfg

@configclass
class GraspObjectSceneCfg(ObjectTableSceneCfg):
    """Scene for the Grasp Object task."""

    # We use a realistically textured generic bowl from the Isaac Nucleus server,
    # and fallback to a primitive cylinder if the URL is unavailable.
    try:
        spawn_cfg = UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_01.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    except Exception:
        spawn_cfg = sim_utils.CylinderCfg(
            radius=0.04,
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.8)),
        )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9], rot=[1, 0, 0, 0]),
        spawn=spawn_cfg,
    )

@configclass
class GraspObjectTerminationsCfg(TerminationsCfg):
    """Terminations for Grasp Object task."""

    success = DoneTerm(
        func=is_object_lifted,
        params={"minimum_height": 1.1, "asset_cfg": SceneEntityCfg("object")},
    )

@configclass
class GraspObjectBaseEnvCfg(PickPlaceGR1T2EnvCfg):
    """Base configuration for Grasp Object environment."""

    scene: GraspObjectSceneCfg = GraspObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    terminations: GraspObjectTerminationsCfg = GraspObjectTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
