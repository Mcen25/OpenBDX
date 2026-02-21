# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for BDX robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_yaw_tracking(
    env: ManagerBasedRLEnv, 
    command_name: str,
    std: float = 0.5
) -> torch.Tensor:
    """Reward for tracking commanded yaw heading direction.
    
    This encourages the robot's forward direction to align with the velocity command direction.
    """
    command = env.command_manager.get_command(command_name)
    
    # Get commanded velocity in world frame
    lin_vel_x_cmd = command[:, 0]
    lin_vel_y_cmd = command[:, 1]
    
    # Compute commanded heading angle
    cmd_heading = torch.atan2(lin_vel_y_cmd, lin_vel_x_cmd)
    
    # Get robot's current yaw (extract from quaternion)
    quat = env.scene["robot"].data.root_quat_w
    yaw = torch.atan2(2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1]), 
                      1.0 - 2.0 * (quat[:, 1]**2 + quat[:, 2]**2))
    
    # Compute heading error
    heading_error = torch.abs(torch.atan2(torch.sin(cmd_heading - yaw), 
                                          torch.cos(cmd_heading - yaw)))
    
    # Only apply when there's significant commanded velocity
    cmd_vel_magnitude = torch.sqrt(lin_vel_x_cmd**2 + lin_vel_y_cmd**2)
    active_mask = cmd_vel_magnitude > 0.1
    
    # Exponential reward for heading tracking
    reward = torch.exp(-heading_error / std) * active_mask.float()
    
    return reward


def joint_target_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_position: float,
    std: float = 0.2
) -> torch.Tensor:
    """Reward for tracking a specific target joint position.
    
    This is useful for gait generation where certain joints should maintain
    specific angles during walking.
    
    Args:
        env: The environment.
        asset_cfg: Scene entity configuration for the joints.
        target_position: Desired joint position in radians.
        std: Standard deviation for exponential reward.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # Compute position error
    pos_error = torch.abs(joint_pos - target_position)
    
    # Exponential reward
    reward = torch.exp(-pos_error / std).mean(dim=1)
    
    return reward
