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

Module API (used by pipeline_ui.py):
    parse_args() -> argparse.Namespace
    start_sim(headless=False) -> SimulationApp
    load_workcell(usd_path) -> None
    load_target_object(object_name) -> None
    find_articulation_root() -> str
    setup_inspection_camera() -> str | None
    build_action_graph(articulation_root, inspection_cam) -> str   # graph path
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UR20_DIR = PROJECT_ROOT / "ur20_description"
# DEFAULT_USD = UR20_DIR / "ur20" / "ur20.usd"
DEFAULT_USD = UR20_DIR / "ur20_with_camera.usd"  # usd의 원점이 달라서 공중에 떠있음
ENV_USD     = UR20_DIR / "environment.usd"
MOUNT_USD   = UR20_DIR / "ur10_mount.usd"
TABLE_USD   = UR20_DIR / "thor_table.usd"

STAGE_PATH = "/World/UR20"
MOUNT_PATH = "/World/Mount"
TABLE_PATH = "/World/Table"
ENV_PATH   = "/World/Environment"
ACTION_GRAPH_PATH = "/ActionGraph"

# 워크셀 치수 (load_environment.py 검증 완료)
MOUNT_HEIGHT = 0.805
TABLE_HEIGHT = 0.630
MOUNT_USD_INTRINSIC_Z = 0.515
TABLE_USD_INTRINSIC_Z = 0.795
MOUNT_XY_SCALE = 2.0
TABLE_USD_BBOX_CENTER_X = 0.270
TABLE_USD_BBOX_CENTER_Y = -0.002
TABLE_TARGET_X = -0.2
TABLE_TARGET_Y = 1.1
ENV_OFFSET = np.array([2.0, 0.0, 0.0])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd-path", type=Path, default=DEFAULT_USD,
                        help=f"Robot USD path (default: {DEFAULT_USD.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--object", type=str, default=None,
                        help="Object name to load workcell (e.g. 'sample')")
    args, _ = parser.parse_known_args(argv)
    return args


def start_sim(headless: bool = False):
    """Create SimulationApp, enable ROS2 bridge, set camera view, return app."""
    from isaacsim import SimulationApp

    config_dict = {"renderer": "RaytracedLighting", "headless": headless}
    simulation_app = SimulationApp(config_dict)

    # These imports are only valid after SimulationApp() exists.
    from isaacsim.core.utils import extensions, viewports

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    viewports.set_camera_view(
        eye=np.array([1.5, 1.5, 1.0]),
        target=np.array([0, 0, 0.5]),
    )
    return simulation_app


def load_workcell(usd_path: Path) -> None:
    """Place environment + mount + table + robot + support cuboid on stage."""
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.utils import prims

    # common/config import deferred until SimulationApp is up.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    prims.create_prim(
        ENV_PATH, "Xform",
        position=ENV_OFFSET,
        usd_path=str(ENV_USD),
    )
    prims.create_prim(
        MOUNT_PATH, "Xform",
        position=np.array([0.0, 0.0, MOUNT_HEIGHT]),
        scale=np.array([MOUNT_XY_SCALE, MOUNT_XY_SCALE,
                        MOUNT_HEIGHT / MOUNT_USD_INTRINSIC_Z]),
        usd_path=str(MOUNT_USD),
    )
    prims.create_prim(
        TABLE_PATH, "Xform",
        position=np.array([
            TABLE_TARGET_X - TABLE_USD_BBOX_CENTER_X,
            TABLE_TARGET_Y - TABLE_USD_BBOX_CENTER_Y,
            TABLE_HEIGHT,
        ]),
        scale=np.array([1.0, 1.0, TABLE_HEIGHT / TABLE_USD_INTRINSIC_Z]),
        usd_path=str(TABLE_USD),
    )
    prims.create_prim(
        STAGE_PATH, "Xform",
        position=np.array([0.0, 0.0, MOUNT_HEIGHT]),
        usd_path=str(usd_path),
    )

    _sup = next(w for w in _config.WALLS if w["name"] == "support")
    VisualCuboid(
        prim_path="/World/Support",
        name="support",
        position=_sup["position"] + np.array([0.0, 0.0, MOUNT_HEIGHT]),
        size=1.0,
        scale=_sup["dimensions"],
        color=np.array([0.5, 0.5, 0.5]),
    )


def load_target_object(object_name: str | None) -> None:
    """Place target object USD (config.TARGET_OBJECT) on the table top."""
    if object_name is None:
        return

    import carb
    from isaacsim.core.utils import prims

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    usd_path = _config.get_mesh_path(object_name, filename="source.usd")
    if not usd_path.exists():
        carb.log_warn(f"Target mesh USD not found: {usd_path}")
        return

    prims.create_prim(
        prim_path=f"/World/{_config.TARGET_OBJECT['name']}",
        prim_type="Xform",
        usd_path=str(usd_path),
        position=np.array([-0.1, 1.1, 0.795]),
        orientation=_config.TARGET_OBJECT["rotation"],
        scale=np.array([1.0, 1.0, 1.0]),
    )


def find_articulation_root() -> str:
    """Locate the ArticulationRootAPI prim under STAGE_PATH; apply API if missing."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(STAGE_PATH):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"Articulation root: {p}")
            return p

    print(f"WARNING: No ArticulationRootAPI found under {STAGE_PATH} — applying to STAGE_PATH")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(STAGE_PATH))
    print(f"Articulation root: {STAGE_PATH}")
    return STAGE_PATH


def setup_inspection_camera() -> str | None:
    """Add an InspectionCamera under SG8S, reusing the existing camera transform."""
    import omni.usd
    from pxr import Gf, UsdGeom

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    stage = omni.usd.get_context().get_stage()

    sg8s_path = None
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(STAGE_PATH):
            continue
        if "SG8S" in prim.GetName():
            sg8s_path = p
            break

    if sg8s_path is None:
        print("WARNING: SG8S prim not found — skipping inspection camera setup")
        return None

    existing_cam = None
    for prim in stage.GetPrimAtPath(sg8s_path).GetAllChildren():
        for desc in [prim] + list(prim.GetAllChildren()):
            if desc.IsA(UsdGeom.Camera):
                existing_cam = desc
                break
        if existing_cam:
            break

    if existing_cam is None:
        print(f"WARNING: No existing Camera found under {sg8s_path} — using identity local pose")
        local = Gf.Matrix4d(1.0)
    else:
        print(f"Reusing transform from existing camera: {existing_cam.GetPath()}")
        xcache = UsdGeom.XformCache()
        cam_world = xcache.GetLocalToWorldTransform(existing_cam)
        sg8s_world = xcache.GetLocalToWorldTransform(stage.GetPrimAtPath(sg8s_path))
        local = cam_world * sg8s_world.GetInverse()

    inspection_cam_path = f"{sg8s_path}/InspectionCamera"
    cam = UsdGeom.Camera.Define(stage, inspection_cam_path)
    UsdGeom.Xformable(cam).ClearXformOpOrder()
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(local))

    # 핀홀 모델: FOV(@d) = d · aperture / focalLength
    cam.GetFocalLengthAttr().Set(float(_config.CAMERA_WORKING_DISTANCE_MM))
    cam.GetHorizontalApertureAttr().Set(float(_config.CAMERA_FOV_WIDTH_MM))
    cam.GetVerticalApertureAttr().Set(float(_config.CAMERA_FOV_HEIGHT_MM))
    cam.GetFocusDistanceAttr().Set(float(_config.CAMERA_WORKING_DISTANCE_MM) * 1e-3)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 5.0))

    print(f"Inspection camera: {inspection_cam_path}")
    return inspection_cam_path


def build_action_graph(articulation_root: str, inspection_cam: str | None) -> str:
    """Create the ROS2 joint-state subscriber + optional camera publishers. Returns graph path."""
    import omni.graph.core as og

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
        ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
        ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
        ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
        ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
    ]
    set_values = [
        ("ArticulationController.inputs:robotPath", articulation_root),
        ("SubscribeJointState.inputs:topicName", "/joint_states"),
    ]

    if inspection_cam is not None:
        create_nodes += [
            ("RP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("RGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("Depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("Info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        set_values += [
            ("RP.inputs:cameraPrim", [inspection_cam]),
            ("RP.inputs:width", _config.CAMERA_PUBLISH_W),
            ("RP.inputs:height", _config.CAMERA_PUBLISH_H),
            ("RGB.inputs:type", "rgb"),
            ("RGB.inputs:topicName", _config.INSPECTION_CAMERA_RGB_TOPIC),
            ("RGB.inputs:frameId", _config.INSPECTION_CAMERA_FRAME_ID),
            ("Depth.inputs:type", "depth"),
            ("Depth.inputs:topicName", _config.INSPECTION_CAMERA_DEPTH_TOPIC),
            ("Depth.inputs:frameId", _config.INSPECTION_CAMERA_FRAME_ID),
            ("Info.inputs:topicName", _config.INSPECTION_CAMERA_INFO_TOPIC),
            ("Info.inputs:frameId", _config.INSPECTION_CAMERA_FRAME_ID),
        ]
        connect += [
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
            {"graph_path": ACTION_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: create_nodes,
                og.Controller.Keys.CONNECT: connect,
                og.Controller.Keys.SET_VALUES: set_values,
            },
        )
    except Exception as e:
        print(e)

    return ACTION_GRAPH_PATH


def main():
    args = parse_args()

    if not args.usd_path.exists():
        sys.exit(
            f"Robot USD not found: {args.usd_path}\n"
            f"Import the URDF via Isaac Sim's URDF Importer GUI and save the USD\n"
            f"to ur20_description/ur20/ur20.usd"
        )

    simulation_app = start_sim(headless=False)

    from isaacsim.core.api import SimulationContext
    simulation_context = SimulationContext(stage_units_in_meters=1.0)

    load_workcell(args.usd_path)
    simulation_app.update()

    load_target_object(args.object)
    simulation_app.update()

    articulation_root = find_articulation_root()
    simulation_app.update()

    inspection_cam = setup_inspection_camera()
    if inspection_cam is not None:
        simulation_app.update()

    build_action_graph(articulation_root, inspection_cam)
    simulation_app.update()

    simulation_context.initialize_physics()
    simulation_context.play()

    while simulation_app.is_running():
        simulation_context.step(render=True)

    simulation_context.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
