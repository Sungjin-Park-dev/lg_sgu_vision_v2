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
    uv run scripts/isaac/ur_ros2_joint_control.py --object sample

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
UR20_DIR = PROJECT_ROOT / "ur20_description"
# DEFAULT_USD = UR20_DIR / "ur20" / "ur20.usd"
DEFAULT_USD = UR20_DIR / "ur20_with_camera.usd" # usd의 원점이 달라서 공중에 떠있음 
ENV_USD     = UR20_DIR / "environment.usd"
MOUNT_USD   = UR20_DIR / "ur10_mount.usd"
TABLE_USD   = UR20_DIR / "thor_table.usd"

STAGE_PATH = "/World/UR20"
MOUNT_PATH = "/World/Mount"
TABLE_PATH = "/World/Table"
ENV_PATH   = "/World/Environment"

# 워크셀 치수 (load_environment.py 검증 완료)
MOUNT_HEIGHT = 0.805    # 로봇 베이스 높이 (m)
TABLE_HEIGHT = 0.630
MOUNT_USD_INTRINSIC_Z = 0.515
TABLE_USD_INTRINSIC_Z = 0.795
MOUNT_XY_SCALE = 2.0
TABLE_USD_BBOX_CENTER_X = 0.270   # USD 로컬 bbox center 보정값
TABLE_USD_BBOX_CENTER_Y = -0.002
TABLE_TARGET_X = -0.2
TABLE_TARGET_Y = 1.1
ENV_OFFSET = np.array([2.0, 0.0, 0.0])

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
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils import extensions, prims, viewports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import config

# Enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")

simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# Set camera view
viewports.set_camera_view(eye=np.array([1.5, 1.5, 1.0]), target=np.array([0, 0, 0.5]))

# ---------------------------------------------------------------------------
# Load environment + workcell (environment.usd + mount + table)
# robot frame == world origin (XY); robot base is elevated by MOUNT_HEIGHT
# ---------------------------------------------------------------------------
prims.create_prim(
    ENV_PATH,
    "Xform",
    position=ENV_OFFSET,
    usd_path=str(ENV_USD),
)

prims.create_prim(
    MOUNT_PATH,
    "Xform",
    position=np.array([0.0, 0.0, MOUNT_HEIGHT]),
    scale=np.array([MOUNT_XY_SCALE, MOUNT_XY_SCALE, MOUNT_HEIGHT / MOUNT_USD_INTRINSIC_Z]),
    usd_path=str(MOUNT_USD),
)

prims.create_prim(
    TABLE_PATH,
    "Xform",
    position=np.array([
        TABLE_TARGET_X - TABLE_USD_BBOX_CENTER_X,
        TABLE_TARGET_Y - TABLE_USD_BBOX_CENTER_Y,
        TABLE_HEIGHT,
    ]),
    scale=np.array([1.0, 1.0, TABLE_HEIGHT / TABLE_USD_INTRINSIC_Z]),
    usd_path=str(TABLE_USD),
)

# Robot — mount 윗면 위에 올림
prims.create_prim(
    STAGE_PATH,
    "Xform",
    position=np.array([0.0, 0.0, MOUNT_HEIGHT]),
    usd_path=str(args.usd_path),
)

# Support — table과 target object 사이를 받치는 막대 (config.WALLS["support"])
# config는 robot frame이라 visual world로 변환: z += MOUNT_HEIGHT
_sup = next(w for w in config.WALLS if w["name"] == "support")
VisualCuboid(
    prim_path="/World/Support",
    name="support",
    position=_sup["position"] + np.array([0.0, 0.0, MOUNT_HEIGHT]),
    size=1.0,
    scale=_sup["dimensions"],
    color=np.array([0.5, 0.5, 0.5]),
)

simulation_app.update()

# ---------------------------------------------------------------------------
# Target object (--object 인자가 있을 때만 로드)
# 위치: 기존 config XY (-0.1, 1.1) + 새 table top 기준 Z = 0.795
# scale=(1,1,1) 로 source.usd 내부 0.01 스케일 덮어씀
# ---------------------------------------------------------------------------
if args.object is not None:
    usd_path = config.get_mesh_path(args.object, filename="source.usd")
    if usd_path.exists():
        prims.create_prim(
            prim_path=f"/World/{config.TARGET_OBJECT['name']}",
            prim_type="Xform",
            usd_path=str(usd_path),
            position=np.array([-0.1, 1.1, 0.795]),
            orientation=config.TARGET_OBJECT["rotation"],
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

# ---------------------------------------------------------------------------
# Inspection camera — SG8S Xform 자식으로 추가. 내부 기존 Camera 의 local
# transform 을 그대로 복사하고 intrinsics 만 config.py 값으로 덮어씀.
# ---------------------------------------------------------------------------
from pxr import Gf, UsdGeom

_sg8s_path = None
for _prim in _stage.Traverse():
    p = str(_prim.GetPath())
    if not p.startswith(STAGE_PATH):
        continue
    if "SG8S" in _prim.GetName():
        _sg8s_path = p
        break

INSPECTION_CAM_PATH = None
if _sg8s_path is None:
    print("WARNING: SG8S prim not found — skipping inspection camera setup")
else:
    # SG8S subtree 에서 기존 Camera prim 찾기 (transform reference 용)
    _existing_cam = None
    for _prim in _stage.GetPrimAtPath(_sg8s_path).GetAllChildren():
        for _desc in [_prim] + list(_prim.GetAllChildren()):
            if _desc.IsA(UsdGeom.Camera):
                _existing_cam = _desc
                break
        if _existing_cam:
            break

    if _existing_cam is None:
        print(f"WARNING: No existing Camera found under {_sg8s_path} — using identity local pose")
        _local = Gf.Matrix4d(1.0)
    else:
        print(f"Reusing transform from existing camera: {_existing_cam.GetPath()}")
        _xcache = UsdGeom.XformCache()
        _cam_world = _xcache.GetLocalToWorldTransform(_existing_cam)
        _sg8s_world = _xcache.GetLocalToWorldTransform(_stage.GetPrimAtPath(_sg8s_path))
        _local = _cam_world * _sg8s_world.GetInverse()

    INSPECTION_CAM_PATH = f"{_sg8s_path}/InspectionCamera"
    _cam = UsdGeom.Camera.Define(_stage, INSPECTION_CAM_PATH)
    UsdGeom.Xformable(_cam).ClearXformOpOrder()
    UsdGeom.Xformable(_cam).AddTransformOp().Set(Gf.Matrix4d(_local))

    # 핀홀 모델: FOV(@d) = d · aperture / focalLength
    # focalLength=WD, aperture=FOV 로 두면 작업거리에서 FOV 가 정확히 맞음
    _cam.GetFocalLengthAttr().Set(float(config.CAMERA_WORKING_DISTANCE_MM))
    _cam.GetHorizontalApertureAttr().Set(float(config.CAMERA_FOV_WIDTH_MM))
    _cam.GetVerticalApertureAttr().Set(float(config.CAMERA_FOV_HEIGHT_MM))
    _cam.GetFocusDistanceAttr().Set(float(config.CAMERA_WORKING_DISTANCE_MM) * 1e-3)
    _cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 5.0))

    print(f"Inspection camera: {INSPECTION_CAM_PATH}")
    simulation_app.update()

# Action Graph — joint state subscribe + (있으면) 카메라 publisher 통합
_create_nodes = [
    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
]
_connect = [
    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
    ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
    ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
]
_set_values = [
    ("ArticulationController.inputs:robotPath", articulation_root_path),
    ("SubscribeJointState.inputs:topicName", "/joint_states"),
]

if INSPECTION_CAM_PATH is not None:
    _create_nodes += [
        ("RP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("RGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("Depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("Info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
    ]
    _set_values += [
        ("RP.inputs:cameraPrim", [INSPECTION_CAM_PATH]),
        ("RP.inputs:width", config.CAMERA_PUBLISH_W),
        ("RP.inputs:height", config.CAMERA_PUBLISH_H),
        ("RGB.inputs:type", "rgb"),
        ("RGB.inputs:topicName", config.INSPECTION_CAMERA_RGB_TOPIC),
        ("RGB.inputs:frameId", config.INSPECTION_CAMERA_FRAME_ID),
        ("Depth.inputs:type", "depth"),
        ("Depth.inputs:topicName", config.INSPECTION_CAMERA_DEPTH_TOPIC),
        ("Depth.inputs:frameId", config.INSPECTION_CAMERA_FRAME_ID),
        ("Info.inputs:topicName", config.INSPECTION_CAMERA_INFO_TOPIC),
        ("Info.inputs:frameId", config.INSPECTION_CAMERA_FRAME_ID),
    ]
    _connect += [
        ("OnPlaybackTick.outputs:tick", "RP.inputs:execIn"),
        ("ROS2Context.outputs:context", "RGB.inputs:context"),
        ("ROS2Context.outputs:context", "Depth.inputs:context"),
        ("ROS2Context.outputs:context", "Info.inputs:context"),
        ("RP.outputs:execOut", "RGB.inputs:execIn"),
        ("RP.outputs:execOut", "Depth.inputs:execIn"),
        ("RP.outputs:execOut", "Info.inputs:execIn"),
        ("RP.outputs:renderProductPath", "RGB.inputs:renderProductPath"),
        ("RP.outputs:renderProductPath", "Depth.inputs:renderProductPath"),
        ("RP.outputs:renderProductPath", "Info.inputs:renderProductPath"),
    ]

try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: _create_nodes,
            og.Controller.Keys.CONNECT: _connect,
            og.Controller.Keys.SET_VALUES: _set_values,
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
