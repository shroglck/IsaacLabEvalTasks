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
from isaaclab_eval_tasks.tasks.manipulation.mdp import is_plant_watered

from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import ObjectTableSceneCfg, PickPlaceGR1T2EnvCfg, TerminationsCfg, EventCfg, ObservationsCfg

@configclass
class WaterPlantSceneCfg(ObjectTableSceneCfg):
    """Scene for the Water Plant task."""

    # Remove the generic 'object' from the base class
    object = None

    # Watering can (e.g. cylinder with offset handle)
    watering_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WateringCan",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
    )

    # Plant (e.g. larger cylinder)
    plant = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plant",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.5, 0.85], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0), # Heavy so it's not easily knocked over
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.2)),
        ),
    )

@configclass
class WaterPlantObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        object_pos = None
        object_rot = None
        object = None

        watering_can_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("watering_can")})
        watering_can_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("watering_can")})
        plant_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("plant")})
        plant_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("plant")})

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()

@configclass
class WaterPlantTerminationsCfg(TerminationsCfg):
    """Terminations for Water Plant task."""

    # Remove object_dropping termination since we replaced 'object'
    object_dropping = None

    success = DoneTerm(
        func=is_plant_watered,
        params={"watering_can_cfg": SceneEntityCfg("watering_can"), "plant_cfg": SceneEntityCfg("plant")},
    )

@configclass
class WaterPlantEventCfg(EventCfg):
    reset_object = None

    reset_watering_can = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("watering_can"),
        },
    )

@configclass
class WaterPlantBaseEnvCfg(PickPlaceGR1T2EnvCfg):
    """Base configuration for Water Plant environment."""

    scene: WaterPlantSceneCfg = WaterPlantSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: WaterPlantObservationsCfg = WaterPlantObservationsCfg()
    terminations: WaterPlantTerminationsCfg = WaterPlantTerminationsCfg()
    events: WaterPlantEventCfg = WaterPlantEventCfg()

    def __post_init__(self):
        super().__post_init__()
