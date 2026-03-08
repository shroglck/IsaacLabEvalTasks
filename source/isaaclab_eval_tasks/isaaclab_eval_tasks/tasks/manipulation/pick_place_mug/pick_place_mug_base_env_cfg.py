# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as task_mdp
from isaaclab_eval_tasks.tasks.manipulation.mdp import is_object_on_plate

from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import ObjectTableSceneCfg, PickPlaceGR1T2EnvCfg, TerminationsCfg, EventCfg, ObservationsCfg

@configclass
class PickPlaceMugSceneCfg(ObjectTableSceneCfg):
    """Scene for the Pick and Place Mug task."""

    # Override object with Mug (cylinder)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
    )

    # Add Plate (flat cylinder)
    plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.45, 0.85], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.1,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
        ),
    )

@configclass
class PickPlaceMugTerminationsCfg(TerminationsCfg):
    """Terminations for Pick and Place Mug task."""

    success = DoneTerm(
        func=is_object_on_plate,
        params={"object_cfg": SceneEntityCfg("object"), "plate_cfg": SceneEntityCfg("plate"), "tolerance": 0.08},
    )

@configclass
class PickPlaceMugBaseEnvCfg(PickPlaceGR1T2EnvCfg):
    """Base configuration for Pick and Place Mug environment."""

    scene: PickPlaceMugSceneCfg = PickPlaceMugSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    terminations: PickPlaceMugTerminationsCfg = PickPlaceMugTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
