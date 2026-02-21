# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

BDX_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/matten/Documents/GitHub/Robotics/OpenBDX/Assets/robot_2/my-robot/robot.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=150.0,
                damping=5.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        joint_pos={
            "Leg1_A": 0.0,
            "Leg1_B": -0.1,
            "Leg1_C": 0.8,
            "Leg1_D": -0.70,
            "Leg1_E": 0.35,
            "Leg2_A": 0.0,
            "Leg2_B": 0.1,
            "Leg2_C": -0.8,
            "Leg2_D": 0.70,
            "Leg2_E": -0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Leg1_.*", "Leg2_.*"],
            effort_limit_sim=200,
            stiffness=150.0,
            damping=5.0,
            armature=0.01,
        ),
    },
)
