# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
UR ROS2 Joint Control standalone script for Isaac Sim.

Loads a UR robot (ur5e or ur20) and creates an Action Graph that subscribes to
/joint_states topic and drives the robot via ArticulationController.

Usage:
    ./python.sh scripts/isaac/ur_ros2_joint_control.py
    ./python.sh scripts/isaac/ur_ros2_joint_control.py --robot ur20

Test with:
    ros2 topic pub /joint_states sensor_msgs/msg/JointState \
        "{name: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
        'wrist_1_joint','wrist_2_joint','wrist_3_joint'],
        position: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}"
"""

import argparse
import sys

import numpy as np
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["ur5e", "ur20"], default="ur5e")
parser.add_argument("--object", type=str, default=None, help="Object name to load workcell (e.g. 'sample')")
args, _ = parser.parse_known_args()

ROBOT_CONFIG = {
    "ur5e": {
        "stage_path": "/World/UR5e",
        "usd_path": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    },
    "ur20": {
        "stage_path": "/World/UR20",
        "usd_path": "/Isaac/Robots/UniversalRobots/ur20/ur20.usd",
    },
}

STAGE_PATH = ROBOT_CONFIG[args.robot]["stage_path"]
USD_PATH = ROBOT_CONFIG[args.robot]["usd_path"]

CONFIG = {"renderer": "RaytracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)

import carb
import omni.graph.core as og
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, prims, viewports
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.objects import VisualCuboid
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "common"))
import config

# Enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Locate Isaac Sim assets folder
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# Set camera view
viewports.set_camera_view(eye=np.array([1.5, 1.5, 1.0]), target=np.array([0, 0, 0.5]))

# Load robot
prims.create_prim(
    STAGE_PATH,
    "Xform",
    position=np.array([0, 0, 0]),
    usd_path=assets_root_path + USD_PATH,
)

simulation_app.update()

# ---------------------------------------------------------------------------
# Load workcell objects from config (table, walls, robot mount, target mesh)
# ---------------------------------------------------------------------------
if args.object is not None:
    # --- Cuboids (table, walls, robot mount) ---
    cuboid_defs = [config.TABLE, config.ROBOT_MOUNT] + config.WALLS
    for c in cuboid_defs:
        VisualCuboid(
            prim_path=f"/World/obstacles/{c['name']}",
            name=c["name"],
            position=c["position"],
            size=1.0,
            scale=c["dimensions"],
            color=np.array([0.8, 0.8, 0.8]),
        )

    # --- Target object mesh (USD) ---
    usd_path = config.get_mesh_path(args.object, filename="source.usd")
    if usd_path.exists():
        pos = config.TARGET_OBJECT["position"]
        rot = config.TARGET_OBJECT["rotation"]  # [w, x, y, z]
        prims.create_prim(
            prim_path=f"/World/{config.TARGET_OBJECT['name']}",
            prim_type="Xform",
            usd_path=str(usd_path),
            position=pos,
            orientation=rot,
            scale=np.array([1.0, 1.0, 1.0]),
        )
    else:
        carb.log_warn(f"Target mesh USD not found: {usd_path}")

    simulation_app.update()

# Create Action Graph with ROS2 SubscribeJointState -> ArticulationController
try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", STAGE_PATH),
                ("SubscribeJointState.inputs:topicName", "/joint_states"),
            ],
        },
    )
except Exception as e:
    print(e)

simulation_app.update()

# Initialize physics and start simulation
simulation_context.initialize_physics()
simulation_context.play()

while simulation_app.is_running():
    simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()
