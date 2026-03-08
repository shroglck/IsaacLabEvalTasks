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
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as task_mdp
from isaaclab_eval_tasks.tasks.manipulation.mdp import nespresso_stage_success

from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import ObjectTableSceneCfg, PickPlaceGR1T2EnvCfg, TerminationsCfg, EventCfg, ObservationsCfg

@configclass
class NespressoSceneCfg(ObjectTableSceneCfg):
    """Scene for the Nespresso task."""

    # Remove generic object
    object = None

    try:
        pod_spawn_cfg = UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/K_Cup/k_cup.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    except Exception:
        pod_spawn_cfg = sim_utils.CylinderCfg(
            radius=0.02,
            height=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.1, 0.1)),
        )

    # Coffee pod (small cylinder)
    pod = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pod",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.85], rot=[1, 0, 0, 0]),
        spawn=pod_spawn_cfg,
    )

    try:
        machine_spawn_cfg = UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Coffee_Maker/coffee_maker.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    except Exception:
        machine_spawn_cfg = sim_utils.CuboidCfg(
            size=(0.15, 0.25, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        )

    # Nespresso Machine (box)
    machine = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Machine",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.5, 0.85], rot=[1, 0, 0, 0]),
        spawn=machine_spawn_cfg,
    )

    # Lever (small cylinder sticking out, assumed rigid body for simplicity of generic primitive representation)
    lever = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Lever",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.5, 1.16], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.01,
            height=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
        ),
    )

@configclass
class NespressoObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        object_pos = None
        object_rot = None
        object = None

        pod_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("pod")})
        pod_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("pod")})
        machine_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("machine")})
        machine_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("machine")})
        lever_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("lever")})
        lever_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("lever")})

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()

@configclass
class NespressoTerminationsCfg(TerminationsCfg):
    """Terminations for Nespresso task."""

    object_dropping = None

    success = DoneTerm(
        func=nespresso_stage_success,
        params={
            "pod_cfg": SceneEntityCfg("pod"),
            "machine_cfg": SceneEntityCfg("machine"),
            "lever_cfg": SceneEntityCfg("lever")
        },
    )

@configclass
class NespressoEventCfg(EventCfg):
    reset_object = None

    reset_pod = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("pod"),
        },
    )

@configclass
class NespressoBaseEnvCfg(PickPlaceGR1T2EnvCfg):
    """Base configuration for Nespresso environment."""

    scene: NespressoSceneCfg = NespressoSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: NespressoObservationsCfg = NespressoObservationsCfg()
    terminations: NespressoTerminationsCfg = NespressoTerminationsCfg()
    events: NespressoEventCfg = NespressoEventCfg()

    def __post_init__(self):
        super().__post_init__()
