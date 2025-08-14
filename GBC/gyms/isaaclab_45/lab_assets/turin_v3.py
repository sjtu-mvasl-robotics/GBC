# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from GBC.utils.base import PROJECT_ROOT_DIR

"""
Configuration for the Turing robot.

Created by: Yifei Yao
Date: 2024-10-28
"""

TURIN_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_ROOT_DIR}/urdf_models/turin_v3/turin_v3.usd",  # Edit this path to the correct path
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
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            "l.*hip.*yaw": 0.0,
            "l.*hip.*roll": 0.0,
            "l.*hip.*pitch": -0.15,
            "l.*knee.*": 0.40,
            "l.*ankle.*pitch": -0.25,
            "l.*ankle.*roll": 0.0,
            "r.*hip.*yaw": 0.0,
            "r.*hip.*roll": 0.0,
            "r.*hip.*pitch": -0.15,
            "r.*knee.*": 0.40,
            "r.*ankle.*pitch": -0.25,
            "r.*ankle.*roll": 0.0,
            ".*waist_roll": 0.0,
            ".*waist_yaw": 0.0,
            ".*head_yaw": 0.0,
            ".*arm_.*pitch": 0.0,
            "l.*arm.*roll":-1.345,
            "r.*arm.*roll": 1.345,
            ".*elbow.*": 0.0,
            # ".*forearm.*": 0.0,
            ".*wrist.*": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*_yaw", ".*hip.*_roll", ".*hip.*_pitch", ".*knee.*"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*hip.*_yaw": 150.0,
                ".*hip.*_roll": 150.0,
                ".*hip.*_pitch": 150.0,
                ".*knee.*": 150.0,
            },
            damping={
                ".*hip.*_yaw": 15.0,
                ".*hip.*_roll": 15.0,
                ".*hip.*_pitch": 15.0,
                ".*knee.*": 15.0,
            },
            armature={
                ".*hip.*_yaw": 0.001,
                ".*hip.*_roll": 0.001,
                ".*hip.*_pitch": 0.001,
                ".*knee.*": 0.001,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle.*"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle.*": 50.0},
            damping={".*_ankle.*": 5.0},
            armature={".*_ankle.*": 0.001},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*arm.*",".*elbow.*",".*wrist.*",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*arm.*": 50.0,
                ".*elbow.*": 50.0,
                # ".*forearm.*": 50.0,
                ".*wrist.*": 50.0,
            },
            damping={
                ".*arm.*": 5.0,
                ".*elbow.*": 5.0,
                # ".*forearm.*": 5.0,
                ".*wrist.*": 5.0,
            },
            armature={
                ".*arm.*": 0.01,
                ".*elbow.*": 0.01,
                # ".*forearm.*": 0.01,
                ".*wrist.*": 0.01,                
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*head.*"],
            effort_limit=50,
            velocity_limit=100.0,
            stiffness={".*head.*": 50.0},
            damping={".*head.*": 5.0},
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[".*waist_roll", ".*waist_yaw"],
            effort_limit=50,
            velocity_limit=100.0,
            stiffness={".*waist_roll": 50.0, ".*waist_yaw": 50.0},
            damping={".*waist_roll": 5.0, ".*waist_yaw": 5.0},
        ),
    },
)
