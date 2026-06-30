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

Prerequisite — USD prepared at workcell/robot/ur20/ur20.usd
(generated via Isaac Sim's URDF Importer GUI with proper articulation root).

Usage:
    uv run scripts/isaac/scene.py --object sample

Test with:
    ros2 topic pub /joint_states sensor_msgs/msg/JointState \
        "{name: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
        'wrist_1_joint','wrist_2_joint','wrist_3_joint'],
        position: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]}"

Module API (used by isaac_pipeline.py):
    parse_args() -> argparse.Namespace
    start_sim(headless=False) -> SimulationApp
    load_workcell(usd_path) -> None
    load_target_object(object_name) -> None
    find_articulation_root() -> str
    set_start_pose(articulation_root, joint_names, positions) -> None
    setup_inspection_camera() -> str | None
    build_action_graph(articulation_root, inspection_cam) -> str   # graph path
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROBOT_DIR = PROJECT_ROOT / "workcell" / "robot"
ENV_DIR   = PROJECT_ROOT / "workcell" / "environment"
# DEFAULT_USD = ROBOT_DIR / "ur20" / "ur20.usd"
DEFAULT_USD = ROBOT_DIR / "ur20_with_camera.usd"  # usd의 원점이 달라서 공중에 떠있음
ENV_USD     = ENV_DIR / "environment.usd"
MOUNT_USD   = ROBOT_DIR / "ur10_mount.usd"
TABLE_USD   = ENV_DIR / "thor_table.usd"

STAGE_PATH = "/World/UR20"
MOUNT_PATH = "/World/Mount"
TABLE_PATH = "/World/Table"
ENV_PATH   = "/World/Environment"
ACTION_GRAPH_PATH = "/ActionGraph"
MOVEIT_GRAPH_PATH = "/MoveItGraph"
CAMERA_MOUNT_NAME = "camera_mount"
CAMERA_OPTICAL_FRAME_NAME = "camera_optical_frame"

# 워크셀 치수 (load_workcell.py 검증 완료)
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
    parser.add_argument("--mode", choices=["sim", "real"], default="sim",
                        help="sim = Isaac-only, no live ROS traffic (default); "
                             "real = mirror /joint_states + publish to the robot")
    parser.add_argument("--pipeline-mode", choices=["inspection", "moveit"],
                        default="inspection",
                        help="Top-level mode (isaac_pipeline.py only). "
                             "inspection = viewpoint/trajectory workflow (sim/real); "
                             "moveit = drive robot from MoveIt via /isaac_joint_commands. "
                             "Ignored by scene.py.")
    args, _ = parser.parse_known_args(argv)
    return args


def start_sim(headless: bool = False, enable_ros_bridge: bool = True):
    """Create SimulationApp, optionally enable ROS2 bridge, set camera view, return app."""
    from isaacsim import SimulationApp

    config_dict = {"renderer": "RaytracedLighting", "headless": headless}
    simulation_app = SimulationApp(config_dict)

    # These imports are only valid after SimulationApp() exists.
    from isaacsim.core.utils import extensions, viewports

    # The action graph that mirrors /joint_states + publishes camera frames
    # needs this extension. We always enable it (eager): the graph is built but
    # left inactive in sim mode, so the toggle just flips its tick — no runtime
    # extension-enable (which is fragile after play() in Isaac 6.0).
    if enable_ros_bridge:
        extensions.enable_extension("isaacsim.ros2.bridge")
    # In GUI mode, also load the OmniGraph editor window so the /ActionGraph is
    # inspectable via Window > Graph Editors > Action Graph. SimulationApp's
    # minimal app does NOT auto-enable editor UI: the graph runtime works (and
    # /ActionGraph shows in Stage), but the editor menu/window are absent until
    # this extension is on.
    if not headless:
        extensions.enable_extension("omni.graph.window.action")
    simulation_app.update()

    viewports.set_camera_view(
        eye=np.array([1.5, 1.5, 1.0]),
        target=np.array([0, 0, 0.5]),
    )
    return simulation_app


def load_workcell(usd_path: Path) -> None:
    """Place environment + mount + table + robot + support cuboid on stage."""
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

    _create_support(_config)


def _create_support(config_module) -> None:
    """Create or replace the visual support using the current collision config."""
    from isaacsim.core.api.objects import VisualCuboid
    from isaacsim.core.utils import prims

    prim_path = "/World/Support"
    if prims.is_prim_path_valid(prim_path):
        prims.delete_prim(prim_path)

    support = next(w for w in config_module.WALLS if w["name"] == "support")
    VisualCuboid(
        prim_path=prim_path,
        name="support",
        position=support["position"] + np.array([0.0, 0.0, MOUNT_HEIGHT]),
        size=1.0,
        scale=support["dimensions"],
        color=np.array([0.5, 0.5, 0.5]),
    )


def load_target_object(object_name: str | None) -> None:
    """Place target object USD (config.TARGET_OBJECT) on the table top.

    Re-callable: if the target prim already exists it is deleted first, so the
    Pipeline UI can swap objects at runtime. The visual is placed at the default
    world pose; the user repositions it afterward with the viewport gizmo.
    """
    if object_name is None:
        return

    import carb
    from isaacsim.core.utils import prims

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    # 물체별 배치를 config 에 반영 → 아래 transform(rotation + world position)이 per-object 가 된다.
    _config.apply_object_placement(object_name)

    usd_path = _config.get_mesh_path(object_name, filename="source.usd")
    if not usd_path.exists():
        carb.log_warn(
            f"Target mesh USD not found: {usd_path}\n"
            f"Build it once: uv run scripts/isaac/usd/build_object_usd.py --object {object_name}"
        )
        return

    # Runtime object swap에서도 support visual을 새 물체 위치에 맞춘다.
    _create_support(_config)

    prim_path = f"/World/{_config.TARGET_OBJECT['name']}"
    if prims.is_prim_path_valid(prim_path):
        prims.delete_prim(prim_path)

    # Reference the USD, then author the LOCAL transform directly via USD ops.
    # We deliberately do NOT pass position/orientation/scale to create_prim:
    # that path sets the *world* pose through XFormPrim, which composes through
    # the physics/Fabric backend while the sim is playing. The boot-time load
    # runs before simulation_context.play() (clean), but the UI "Load Object"
    # button runs while playing → the object spawned with a wrong rotation.
    # Authoring local USD ops is play-state independent, so boot and button
    # loads are now identical. /World is at the origin, so local == world here.
    prims.create_prim(prim_path=prim_path, prim_type="Xform", usd_path=str(usd_path))

    import omni.usd
    from pxr import Gf, UsdGeom

    q = _config.TARGET_OBJECT["rotation"]  # (w, x, y, z)
    wp = _config.target_object_world_position()  # robot frame → world (z += MOUNT_HEIGHT)
    M = Gf.Matrix4d()
    M.SetTransform(
        Gf.Rotation(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3]))),
        Gf.Vec3d(float(wp[0]), float(wp[1]), float(wp[2])),
    )
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(M)


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


def set_start_pose(articulation_root: str, joint_names, positions) -> None:
    """Stand the articulation at `positions` (radians, in `joint_names` order).

    Must be called after SimulationContext.play() (and at least one step) so the
    physics view is bound. It does two things:
      1. set_joint_positions — teleports the joint *state* to the pose, and
      2. apply_action       — sets the PD drive *targets* to the same pose.
    Both are required: set_joint_positions only teleports, so without (2) the
    position drives would pull the joints back to 0 on the next physics step.
    Joints are matched by name via get_dof_index, so the articulation's internal
    DOF ordering is handled automatically.
    """
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    art = SingleArticulation(prim_path=articulation_root)
    art.initialize()
    indices = np.array([art.get_dof_index(n) for n in joint_names], dtype=np.int32)
    q = np.asarray(positions, dtype=np.float64)
    art.set_joint_positions(q, joint_indices=indices)
    art.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=q, joint_indices=indices)
    )
    print(f"Start pose set: {np.rad2deg(q).round(1).tolist()} deg")


def setup_inspection_camera() -> str | None:
    """Add an InspectionCamera under the camera optical frame.

    Prefer the local camera_optical_frame prim. Fall back to camera_mount, then
    the legacy SG8S prim so older USD files still publish camera topics while
    the asset is being migrated.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    stage = omni.usd.get_context().get_stage()

    camera_frame_path = None
    camera_mount_path = None
    legacy_sg8s_path = None
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(STAGE_PATH):
            continue
        if prim.GetName() == CAMERA_OPTICAL_FRAME_NAME:
            camera_frame_path = p
            break
        if prim.GetName() == CAMERA_MOUNT_NAME:
            camera_mount_path = p
        if legacy_sg8s_path is None and "SG8S" in prim.GetName():
            legacy_sg8s_path = p

    camera_frame_path = camera_frame_path or camera_mount_path or legacy_sg8s_path
    if camera_frame_path is None:
        print(
            f"WARNING: {CAMERA_OPTICAL_FRAME_NAME} prim not found — "
            "skipping inspection camera setup"
        )
        return None

    existing_cam = None
    for prim in stage.GetPrimAtPath(camera_frame_path).GetAllChildren():
        for desc in [prim] + list(prim.GetAllChildren()):
            if desc.IsA(UsdGeom.Camera):
                existing_cam = desc
                break
        if existing_cam:
            break

    if existing_cam is None:
        print(
            f"WARNING: No existing Camera found under {camera_frame_path} — "
            "using default local pose"
        )
        frame_prim = stage.GetPrimAtPath(camera_frame_path)
        if frame_prim.GetName() == CAMERA_OPTICAL_FRAME_NAME:
            # The planner's camera_optical_frame uses +Z as the viewing axis.
            # UsdGeom.Camera looks down local -Z, so rotate the camera 180 deg
            # about Y to align its optical axis with the planner frame.
            local = Gf.Matrix4d(
                -1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, -1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0,
            )
        else:
            local = Gf.Matrix4d(1.0)
    else:
        print(f"Reusing transform from existing camera: {existing_cam.GetPath()}")
        xcache = UsdGeom.XformCache()
        cam_world = xcache.GetLocalToWorldTransform(existing_cam)
        frame_world = xcache.GetLocalToWorldTransform(stage.GetPrimAtPath(camera_frame_path))
        local = cam_world * frame_world.GetInverse()

    inspection_cam_path = f"{camera_frame_path}/InspectionCamera"
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
    import omni.usd
    from isaacsim.core.utils import prims

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    # Idempotency guard: ACTION_GRAPH_PATH is fixed, so re-running this would try
    # to CREATE_NODES onto an existing graph (duplicate-node error). Tear any
    # existing graph down first so the function is safe to call more than once.
    stage = omni.usd.get_context().get_stage()
    if stage is not None and stage.GetPrimAtPath(ACTION_GRAPH_PATH).IsValid():
        prims.delete_prim(ACTION_GRAPH_PATH)

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


def build_moveit_graph(articulation_root: str) -> str:
    """Create the MoveIt(cuMotion) bridge graph. Returns graph path.

    Mirrors build_action_graph but wired for the isaac_ros_cumotion / MoveIt
    TopicBasedSystem convention (ur.ros2_control.xacro):
      - subscribe MOVEIT_JOINT_COMMANDS_TOPIC (/isaac_joint_commands) → drive robot
      - publish   MOVEIT_JOINT_STATES_TOPIC   (/isaac_joint_states)   ← robot state

    Built as a SEPARATE graph from /ActionGraph so the two can be gated
    independently (the pipeline UI keeps only one ticking at a time).
    """
    import omni.graph.core as og
    import omni.usd
    from isaacsim.core.utils import prims

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
    import config as _config

    # Idempotency guard (see build_action_graph).
    stage = omni.usd.get_context().get_stage()
    if stage is not None and stage.GetPrimAtPath(MOVEIT_GRAPH_PATH).IsValid():
        prims.delete_prim(MOVEIT_GRAPH_PATH)

    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        # ROS → Isaac: MoveIt-executed trajectory commands drive the articulation.
        ("SubscribeJointCommand", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        # Isaac → ROS: publish current joint state for ros2_control feedback.
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        # Isaac → ROS: publish /clock so use_sim_time consumers (MoveIt,
        # controller_manager) advance. Without this, ros2_control stalls waiting
        # for sim time and controllers never activate.
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "SubscribeJointCommand.inputs:execIn"),
        ("SubscribeJointCommand.outputs:execOut", "ArticulationController.inputs:execIn"),
        ("ROS2Context.outputs:context", "SubscribeJointCommand.inputs:context"),
        ("SubscribeJointCommand.outputs:jointNames", "ArticulationController.inputs:jointNames"),
        ("SubscribeJointCommand.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
        ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
    ]
    set_values = [
        ("ArticulationController.inputs:robotPath", articulation_root),
        ("SubscribeJointCommand.inputs:topicName", _config.MOVEIT_JOINT_COMMANDS_TOPIC),
        ("PublishJointState.inputs:targetPrim", [articulation_root]),
        ("PublishJointState.inputs:topicName", _config.MOVEIT_JOINT_STATES_TOPIC),
        ("PublishClock.inputs:topicName", "/clock"),
        # Keep sim time monotonic across Stop/Play (default True resets it to 0).
        # A backward /clock jump confuses MoveIt/controller_manager (use_sim_time)
        # and makes them take a long time to re-sync after Play; monotonic time
        # lets them reconnect immediately.
        ("ReadSimTime.inputs:resetOnStop", False),
    ]

    try:
        og.Controller.edit(
            {"graph_path": MOVEIT_GRAPH_PATH, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: create_nodes,
                og.Controller.Keys.CONNECT: connect,
                og.Controller.Keys.SET_VALUES: set_values,
            },
        )
    except Exception as e:
        print(e)

    return MOVEIT_GRAPH_PATH


def main():
    args = parse_args()

    if not args.usd_path.exists():
        sys.exit(
            f"Robot USD not found: {args.usd_path}\n"
            f"Import the URDF via Isaac Sim's URDF Importer GUI and save the USD\n"
            f"to workcell/robot/ur20/ur20.usd"
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
