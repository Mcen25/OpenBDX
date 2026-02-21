# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import BDXRoughEnvCfg


@configclass
class BDXFlatEnvCfg(BDXRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.dof_torques_l2.weight = -1.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["Leg.*_[CD]"]
        )
        self.rewards.base_yaw_tracking.weight = 2.0
        self.rewards.joint_gait_target.weight = 0.5
        self.rewards.joint_gait_target.params["target_position"] = -0.8
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)


class BDXFlatEnvCfg_PLAY(BDXFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
