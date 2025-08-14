# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`GR1_CFG`:Fourier GR1 robot with DC motor model for the legs


"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from GBC.utils.base import PROJECT_ROOT_DIR


##
# Configuration - Actuators.
##
GR1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/yifei/codespace/GBC_45/robots_usd/fourier_gr1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            # ".*_hip_pitch_joint": -0.20,
            # ".*_hip_roll_joint": 0.00,
            # ".*_hip_yaw_joint": 0.00,
            # ".*_knee_pitch_joint": 0.42,
            # ".*_ankle_pitch_joint": -0.23,
            # ".*_ankle_roll_joint": 0.00,
            # "waist_.*_joint":0.00,
        
            
            # "left_elbow_joint": 0.00,
            # "left_shoulder_roll_joint": 0.16,
            # "left_shoulder_pitch_joint": 0.35,
            # "left_shoulder_yaw_joint": 0.00,
            # "left_wrist_roll_joint":0.00,
            # "left_wrist_pitch_joint":0.00,
            # "left_wrist_yaw_joint":0.00,
            
            # "right_elbow_joint": 0.00,
            # "right_shoulder_roll_joint": -0.16,
            # "right_shoulder_pitch_joint": 0.35,
            # "right_shoulder_yaw_joint": 0.00,
            # "right_wrist_roll_joint":0.00,
            # "right_wrist_pitch_joint":0.00,
            # "right_wrist_yaw_joint":0.00,
            ".*": 0.0,
            
            
            # "left_one_joint": 1.0,
            # "right_one_joint": -1.0,
            # "left_two_joint": 0.52,
            # "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint", 
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_pitch_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_pitch_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "waists": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=150,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                
                ".*_elbow_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)