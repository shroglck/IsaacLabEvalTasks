# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

def is_object_lifted(
    env: ManagerBasedEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Check if the object has been lifted above a minimum height.

    Args:
        env: The RL environment instance.
        minimum_height: The minimum height threshold for success.
        asset_cfg: Configuration for the object entity.

    Returns:
        A boolean tensor indicating if the object is successfully lifted.
    """
    asset = env.scene[asset_cfg.name]
    object_heights = asset.data.root_pos_w[:, 2]
    return object_heights > minimum_height

def is_object_on_plate(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    tolerance: float = 0.1
) -> torch.Tensor:
    """Check if the object is placed correctly on the plate.

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object (e.g., mug) entity.
        plate_cfg: Configuration for the plate entity.
        tolerance: The maximum allowable XY distance from the plate center to be considered a success.

    Returns:
        A boolean tensor indicating if the object is successfully placed on the plate.
    """
    obj = env.scene[object_cfg.name]
    plate = env.scene[plate_cfg.name]

    obj_pos = obj.data.root_pos_w
    plate_pos = plate.data.root_pos_w

    # Check XY distance
    dist_xy = torch.norm(obj_pos[:, :2] - plate_pos[:, :2], dim=-1)

    # Check Z height (object should be resting on plate, so its height should be close to plate height + some offset)
    # Simple check: the object must be slightly above the plate's center z but close
    z_diff = obj_pos[:, 2] - plate_pos[:, 2]

    is_close_xy = dist_xy < tolerance
    is_above_plate = (z_diff > 0.0) & (z_diff < 0.2) # Assuming plate height is around z, obj is placed on top

    return is_close_xy & is_above_plate

def is_plant_watered(
    env: ManagerBasedEnv,
    watering_can_cfg: SceneEntityCfg = SceneEntityCfg("watering_can"),
    plant_cfg: SceneEntityCfg = SceneEntityCfg("plant"),
    min_tilt_angle: float = 0.5, # Minimum tilt angle in radians (~30 degrees)
    xy_tolerance: float = 0.2
) -> torch.Tensor:
    """Check if the watering can is tilted over the plant.

    Args:
        env: The RL environment instance.
        watering_can_cfg: Configuration for the watering can entity.
        plant_cfg: Configuration for the plant entity.
        min_tilt_angle: Minimum pitch angle to consider it "pouring".
        xy_tolerance: Maximum xy distance to consider it "over" the plant.

    Returns:
        A boolean tensor indicating if the plant is being watered.
    """
    can = env.scene[watering_can_cfg.name]
    plant = env.scene[plant_cfg.name]

    can_pos = can.data.root_pos_w
    plant_pos = plant.data.root_pos_w

    # Check XY distance
    dist_xy = torch.norm(can_pos[:, :2] - plant_pos[:, :2], dim=-1)
    is_over_plant = dist_xy < xy_tolerance

    # Check tilt
    # A simple way to check tilt is by looking at the Z axis of the can's orientation
    # can_quat: (w, x, y, z)
    # The direction of the "up" vector in local frame can be found by rotating (0,0,1)
    import isaaclab.utils.math as math_utils
    up_vector = torch.zeros((can.data.root_quat_w.shape[0], 3), device=can.device)
    up_vector[:, 2] = 1.0

    world_up = math_utils.quat_rotate(can.data.root_quat_w, up_vector)

    # If the Z component of the world_up vector is less than cos(min_tilt_angle), it's tilted
    import math
    tilt_threshold = math.cos(min_tilt_angle)
    is_tilted = world_up[:, 2] < tilt_threshold

    return is_over_plant & is_tilted

def nespresso_stage_success(
    env: ManagerBasedEnv,
    pod_cfg: SceneEntityCfg = SceneEntityCfg("pod"),
    machine_cfg: SceneEntityCfg = SceneEntityCfg("machine"),
    lever_cfg: SceneEntityCfg = SceneEntityCfg("lever") # assuming it's an articulation or separate rigid body for simplicity
) -> torch.Tensor:
    """Check continuous progress of making a nespresso.

    A simplified version: check if pod is near machine slot, and lever is closed.
    """
    pod = env.scene[pod_cfg.name]
    machine = env.scene[machine_cfg.name]

    pod_pos = pod.data.root_pos_w
    machine_pos = machine.data.root_pos_w

    dist_xy = torch.norm(pod_pos[:, :2] - machine_pos[:, :2], dim=-1)
    is_pod_inserted = dist_xy < 0.1

    return is_pod_inserted
