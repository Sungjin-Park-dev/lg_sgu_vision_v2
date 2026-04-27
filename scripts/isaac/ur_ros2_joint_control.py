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

Loads our local UR20 (camera-attached) and creates an Action Graph that
subscribes to /joint_states and drives the robot via ArticulationController.

Prerequisite — USD prepared at ur20_description/ur20/ur20.usd
(generated via Isaac Sim's URDF Importer GUI with proper articulation root).

Usage:
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python \
        scripts/isaac/ur_ros2_joint_control.py
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python \
        scripts/isaac/ur_ros2_joint_control.py --object sample

Test with:
    ros2 topic pub /joint_states sensor_msgs/msg/JointState \
        "{name: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
        'wrist_1_joint','wrist_2_joint','wrist_3_joint'],
        position: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_USD = PROJECT_ROOT / "ur20_description" / "ur20" / "ur20.usd"
STAGE_PATH = "/World/UR20"

parser = argparse.ArgumentParser()
parser.add_argument("--usd-path", type=Path, default=DEFAULT_USD,
                    help=f"Robot USD path (default: {DEFAULT_USD.relative_to(PROJECT_ROOT)})")
parser.add_argument("--object", type=str, default=None,
                    help="Object name to load workcell (e.g. 'sample')")
args, _ = parser.parse_known_args()

if not args.usd_path.exists():
    sys.exit(
        f"Robot USD not found: {args.usd_path}\n"
        f"Import the URDF via Isaac Sim's URDF Importer GUI and save the USD\n"
        f"to ur20_description/ur20/ur20.usd"
    )

CONFIG = {"renderer": "RaytracedLighting", "headless": False}

simulation_app = SimulationApp(CONFIG)

import carb
import omni.graph.core as og
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, prims, viewports
from isaacsim.core.api.objects import VisualCuboid
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import config

# Enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Set camera view
viewports.set_camera_view(eye=np.array([1.5, 1.5, 1.0]), target=np.array([0, 0, 0.5]))

# Load robot from local USD
prims.create_prim(
    STAGE_PATH,
    "Xform",
    position=np.array([0, 0, 0]),
    usd_path=str(args.usd_path),
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

# Locate articulation root inside the loaded USD
from pxr import UsdPhysics
import omni.usd
_stage = omni.usd.get_context().get_stage()
articulation_root_path = None
for _prim in _stage.Traverse():
    p = str(_prim.GetPath())
    if not p.startswith(STAGE_PATH):
        continue
    if _prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_root_path = p
        break
if articulation_root_path is None:
    print(f"WARNING: No ArticulationRootAPI found under {STAGE_PATH} — applying to STAGE_PATH")
    UsdPhysics.ArticulationRootAPI.Apply(_stage.GetPrimAtPath(STAGE_PATH))
    articulation_root_path = STAGE_PATH
print(f"Articulation root: {articulation_root_path}")
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
                ("ArticulationController.inputs:robotPath", articulation_root_path),
                ("SubscribeJointState.inputs:topicName", "/joint_states"),
            ],
        },
    )
except Exception as e:
    print(e)

simulation_app.update()

# Open Action Graph editor window
import omni.kit.app
import omni.kit.commands
_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ext_manager.set_extension_enabled_immediate("omni.graph.window.action", True)
simulation_app.update()
omni.kit.commands.execute("OpenWindow", window_name="Action Graph")
simulation_app.update()

# Initialize physics and start simulation
simulation_context.initialize_physics()
simulation_context.play()

while simulation_app.is_running():
    simulation_context.step(render=True)

simulation_context.stop()
simulation_app.close()
