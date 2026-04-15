#!/usr/bin/env python3
"""
사전에 계산된 충돌 없는 궤적을 Isaac Sim으로 실행

사용법:
    omni_python scripts/3_simulation.py \
        --object sample \
        --num_viewpoints 163

    # Tilt 궤적
    omni_python scripts/3_simulation.py \
        --object sample \
        --num_viewpoints 163 \
        --tilt
"""

# ============================================================================
# Section 1: Imports
# ============================================================================

# Standard Library
import argparse
import csv
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Tuple, Optional, Dict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Third Party
import numpy as np

# Isaac Sim (must initialize SimulationApp early)
try:
    import isaacsim
except ImportError:
    pass

from isaacsim.simulation_app import SimulationApp

# ============================================================================
# Section 2: CLI Argument Parsing (before SimulationApp)
# ============================================================================

parser = argparse.ArgumentParser(
    description="Simulate robot trajectory in Isaac Sim (educational version)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# Required arguments
parser.add_argument(
    "--object",
    type=str,
    required=True,
    help="Object name for auto-path generation (e.g., 'sample', 'glass')"
)
parser.add_argument(
    "--num_viewpoints",
    type=int,
    required=True,
    help="Number of viewpoints"
)

# Optional arguments
parser.add_argument(
    "--robot",
    type=str,
    default="ur20_with_camera.yml"
)

parser.add_argument(
    "--tilt",
    action="store_true",
    help="Use tilt trajectory (tilt_trajectory.csv) instead of regular trajectory",
    default=False
)

parser.add_argument(
    "--trajectory-file",
    type=str,
    default=None,
    help="Custom trajectory filename (overrides --tilt if specified)"
)

parser.add_argument(
    "--headless",
    type=str,
    default=None,
    help="Run headless: one of [native, websocket]"
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Visualize robot collision spheres",
    default=False
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode: visualize target waypoint positions as green points"
)

args = parser.parse_args()

# Initialize SimulationApp (must happen before importing other Isaac Sim modules)
simulation_app = SimulationApp({
    "headless": args.headless is not None,
    "width": "1280",
    "height": "720",
})

# ============================================================================
# Isaac Sim Component Imports (after SimulationApp)
# ============================================================================
from omni.isaac.core import World
from omni.isaac.core.objects import sphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from pxr import UsdPhysics

try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    from isaacsim.util.debug_draw import _debug_draw

from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage

try:
    from isaacsim.core.api.materials import OmniGlass, OmniPBR
except ImportError:
    from omni.isaac.core.materials import OmniGlass, OmniPBR

# Isaac Sim URDF importer
ISAAC_SIM_23 = False
ISAAC_SIM_45 = False
try:
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    try:
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+
        ISAAC_SIM_45 = True
    ISAAC_SIM_23 = True

# ============================================================================
# CuRobo Imports
# ============================================================================
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.types.math import quat_multiply
from curobo.util.logger import log_warn
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    get_assets_path,
    get_filename,
    get_path_of_dir,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# ============================================================================
# Local Imports
# ============================================================================
import common.config as config


# ============================================================================
# Section 3: Inlined Utilities from common/
# ============================================================================

# ----------------------------------------------------------------------------
# Section 3.1: CLI Utils (from common/cli_utils.py)
# ----------------------------------------------------------------------------

def print_section_header(
    title: str,
    width: int = 70,
    char: str = '=',
    newline_before: bool = True
) -> None:
    """
    Print a formatted section header

    Args:
        title: Section title text
        width: Total width of the header (default: 70)
        char: Character to use for borders (default: '=')
        newline_before: Add newline before header (default: True)

    Example:
        >>> print_section_header("INITIALIZATION")

        ======================================================================
        INITIALIZATION
        ======================================================================
    """
    if newline_before:
        print()
    print(char * width)
    print(title)
    print(char * width)


def print_key_value(
    key: str,
    value: any,
    indent: int = 2,
    width: int = 35
) -> None:
    """
    Print a key-value pair with consistent formatting

    Args:
        key: Key name
        value: Value to print
        indent: Number of spaces to indent (default: 2)
        width: Width for key column (default: 35)

    Example:
        >>> print_key_value("Total waypoints", 150)
          Total waypoints:                   150
    """
    indent_str = ' ' * indent
    print(f"{indent_str}{key:<{width}}: {value}")


def print_success(message: str, indent: int = 0) -> None:
    """
    Print a success message with checkmark

    Args:
        message: Success message
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_success("Trajectory saved")
         Trajectory saved
    """
    indent_str = ' ' * indent
    print(f"{indent_str} {message}")


def print_warning(message: str, indent: int = 0) -> None:
    """
    Print a warning message

    Args:
        message: Warning message
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_warning("Mesh coordinates may be incorrect")
        � WARNING: Mesh coordinates may be incorrect
    """
    indent_str = ' ' * indent
    print(f"{indent_str}� WARNING: {message}")


def print_error(message: str, indent: int = 0) -> None:
    """
    Print an error message

    Args:
        message: Error message
        indent: Number of spaces to indent (default: 0)

    Example:
        >>> print_error("File not found")
         ERROR: File not found
    """
    indent_str = ' ' * indent
    print(f"{indent_str} ERROR: {message}")


# ----------------------------------------------------------------------------
# Section 3.2: Data I/O (from common/data_io.py)
# ----------------------------------------------------------------------------

def load_trajectory_csv(
    csv_path: str,
    joint_prefix: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load joint trajectory from CSV file

    Args:
        csv_path: Path to CSV file
        joint_prefix: Optional prefix to filter joint columns (e.g., "ur20-")
                     If None, uses all columns with "joint" in name

    Returns:
        trajectory: (N, n_joints) array of joint angles
        joint_names: List of joint column names

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If no joint columns found or invalid data

    Example:
        >>> trajectory, joint_names = load_trajectory_csv(
        ...     "data/trajectory/path.csv",
        ...     joint_prefix="ur20-"
        ... )
        >>> print(trajectory.shape)
        (150, 6)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory file not found: {csv_path}")

    joint_angles = []
    joint_names = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError(f"CSV file has no headers: {csv_path}")

        # Extract joint column names
        if joint_prefix:
            joint_names = [h for h in headers if h.startswith(joint_prefix)]
        else:
            joint_names = [h for h in headers if 'joint' in h.lower()]

        if not joint_names:
            raise ValueError(
                f"No joint columns found in CSV. Headers: {headers}\n"
                f"Use joint_prefix parameter to specify joint column prefix."
            )

        # Read trajectory data
        for row in reader:
            try:
                config_vals = [float(row[joint_name]) for joint_name in joint_names]
                joint_angles.append(config_vals)
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Error parsing row in {csv_path}: {e}\n"
                    f"Joint names: {joint_names}"
                )

    if not joint_angles:
        raise ValueError(f"No trajectory data found in {csv_path}")

    trajectory = np.array(joint_angles, dtype=np.float64)

    print(f"Loaded trajectory: {len(trajectory)} waypoints, {len(joint_names)} joints")
    print(f"Joint names: {joint_names}")

    return trajectory, joint_names


# ----------------------------------------------------------------------------
# Section 3.3: World Setup (from common/world_setup.py)
# ----------------------------------------------------------------------------

def setup_collision_world(
    table: Dict,
    walls: List[Dict],
    robot_mount: Dict,
    target_object: Dict,
    target_mesh_path: Optional[str] = None,
) -> WorldConfig:
    """
    Setup collision world configuration with all obstacles

    Args:
        table: Table config dict with 'name', 'position', 'dimensions' keys
        walls: List of wall dicts with 'name', 'position', 'dimensions' keys
        robot_mount: Robot mount config dict with 'name', 'position', 'dimensions' keys
        target_object: Target object config dict with 'name', 'position', 'rotation' keys
        target_mesh_path: Optional path to target object mesh file for collision

    Returns:
        WorldConfig containing all configured obstacles
    """
    # Load base world config (table)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose = list(table["position"]) + [1, 0, 0, 0]
    world_cfg_table.cuboid[0].dims = list(table["dimensions"])
    world_cfg_table.cuboid[0].name = table["name"]

    # Add wall cuboids
    wall_cuboids = []
    for wall in walls:
        wall_dict = {
            "table": {
                "dims": wall["dimensions"].tolist(),
                "pose": list(wall["position"]) + [1, 0, 0, 0]
            }
        }
        wall_cfg = WorldConfig.from_dict({"cuboid": wall_dict})
        wall_cfg.cuboid[0].name = wall["name"]
        wall_cuboids.extend(wall_cfg.cuboid)

    # Add robot mount cuboid
    robot_mount_cuboid_dict = {
        "table": {
            "dims": robot_mount["dimensions"].tolist(),
            "pose": list(robot_mount["position"]) + [1, 0, 0, 0]
        }
    }
    robot_mount_cfg = WorldConfig.from_dict({"cuboid": robot_mount_cuboid_dict})
    robot_mount_cfg.cuboid[0].name = robot_mount["name"]

    # Combine all cuboids
    all_cuboids = (
        world_cfg_table.cuboid +
        wall_cuboids +
        robot_mount_cfg.cuboid
    )

    # Ensure all cuboid poses are consistent Python lists (avoid numpy array mixing)
    for cuboid in all_cuboids:
        if hasattr(cuboid, 'pose') and cuboid.pose is not None:
            cuboid.pose = list(cuboid.pose)
        if hasattr(cuboid, 'dims') and cuboid.dims is not None:
            cuboid.dims = list(cuboid.dims)

    # Add target object mesh if provided
    meshes = []
    if target_mesh_path is not None:
        target_mesh = Mesh(
            name=target_object["name"],
            file_path=target_mesh_path,
            pose=list(target_object["position"]) + list(target_object["rotation"]),
        )
        meshes.append(target_mesh)

    world_cfg = WorldConfig(cuboid=all_cuboids, mesh=meshes if meshes else None)

    return world_cfg

# ----------------------------------------------------------------------------
# Section 3.4: Simulation Helper (from common/simulation_helper.py)
# ----------------------------------------------------------------------------

def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    """
    Enable required Isaac Sim extensions

    Args:
        simulation_app: SimulationApp instance
        headless_mode: Optional headless mode ("native" or "websocket")

    Returns:
        True when complete
    """
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


def add_robot_to_scene(
    robot_config: dict,
    my_world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.ndarray = np.array([0, 0, 0]),
    initialize_world: bool = True,
):
    """
    Add robot to scene using URDF import

    Handles Isaac Sim version differences (2023.1 vs 4.5+).

    Args:
        robot_config: Robot configuration dictionary from YAML
        my_world: Isaac Sim World instance
        load_from_usd: Load from USD instead of URDF (default: False)
        subroot: Subroot path for robot (default: "")
        robot_name: Name for robot (default: "robot")
        position: Robot base position (default: [0, 0, 0])
        initialize_world: Initialize world after adding robot (default: True)

    Returns:
        Tuple of (robot, robot_path)
    """
    urdf_interface = _urdf.acquire_urdf_interface()

    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path contains the path to urdf
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)

    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model
        dest_path = join_path(
            root_path, get_filename(file_name, remove_extension=True) + "_temp.usd"
        )

        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:
        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    return robot, robot_path


# ============================================================================
# Section 4: World Initialization
# ============================================================================

@dataclass
class WorldState:
    """Encapsulates Isaac Sim world state"""
    world: World
    target_object_prim: XFormPrim
    robot: any
    idx_list: List[int]
    ik_solver: IKSolver


def create_world() -> World:
    """Create Isaac Sim world"""
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    return my_world


def setup_robot(my_world: World, robot_config_file: str) -> dict:
    """Setup robot in the world

    Returns:
        dict with keys: robot, idx_list, robot_prim_path, robot_cfg
    """
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config_file))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(
        robot_config=robot_cfg,
        my_world=my_world,
        position=np.array([0.0, 0.0, 0.0]),
    )

    idx_list = [robot.get_dof_index(x) for x in j_names]
    robot.set_joint_positions(default_config, idx_list)

    return {
        'robot': robot,
        'idx_list': idx_list,
        'robot_prim_path': robot_prim_path,
        'robot_cfg': robot_cfg,
    }


def setup_object_from_mesh(
    my_world: World,
    mesh_path: str,
    usd_helper: UsdHelper
) -> Optional[XFormPrim]:
    """Setup inspection object using mesh file"""
    if not os.path.exists(mesh_path):
        print_warning(f"Mesh file not found: {mesh_path}")
        print_warning("Simulation will run without object mesh")
        return None

    print_section_header("ADDING OBJECT MESH TO STAGE", width=70)
    print_key_value("Mesh file", mesh_path)
    print_key_value("Position", config.TARGET_OBJECT["position"])
    print()

    usd_helper.load_stage(my_world.stage)

    target_object_mesh = Mesh(
        name=config.TARGET_OBJECT["name"],
        file_path=mesh_path,
        pose=list(config.TARGET_OBJECT["position"]) + list(config.TARGET_OBJECT["rotation"]),
        color=[1.0, 0.1, 0.1, 0.95]
    )

    target_object_path = usd_helper.add_mesh_to_stage(
        obstacle=target_object_mesh,
        base_frame="/World"
    )

    print_key_value("Object prim path", target_object_path)

    target_object_prim = XFormPrim(target_object_path)

    return target_object_prim


def setup_camera(robot_prim_path: str, my_world: World):
    """Setup camera mounted on robot end-effector"""
    tool_prim_path = robot_prim_path + "/tool0"
    camera_prim_path = tool_prim_path + "/mounted_camera"

    camera = Camera(
        prim_path=camera_prim_path,
        frequency=20,
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1, 0, 0, 0]),
        resolution=(256, 256),
    )

    # Camera specifications
    camera.set_focal_length(38.0 / 1e3)
    camera.set_focus_distance(110.0 / 1e3)
    camera.set_horizontal_aperture(14.13 / 1e3)
    camera.set_vertical_aperture(10.35 / 1e3)
    camera.set_clipping_range(10/1e3, 100/1e3)
    camera.set_local_pose(
        np.array([0.0, 0.0, 0.0]),
        euler_angles_to_quats(np.array([0, 180, 0]), degrees=True),
        camera_axes="usd"
    )
    my_world.scene.add(camera)

    return camera


def setup_collision_checker(
    my_world: World,
    robot_state: dict,
    mesh_path: Optional[str] = None,
) -> IKSolver:
    """Setup collision checker and IK solver"""
    usd_helper = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = robot_state['robot_cfg']
    robot_prim_path = robot_state['robot_prim_path']

    # Setup world collision configuration using inlined utility
    world_cfg = setup_collision_world(
        table=config.TABLE,
        walls=config.WALLS,
        robot_mount=config.ROBOT_MOUNT,
        target_object=config.TARGET_OBJECT,
        target_mesh_path=mesh_path,
    )

    # Add ground mesh (positioned below ground)
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    # Combine cuboids and mesh obstacles
    world_cfg = WorldConfig(cuboid=world_cfg.cuboid, mesh=world_cfg1.mesh)

    # Create IK solver (needed for sphere visualization)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": 4, "mesh": 1},
    )
    ik_solver = IKSolver(ik_config)

    # Setup world in USD
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane(z_position=-0.5)

    # Get obstacles from stage
    obstacles = usd_helper.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,
            "/World/defaultGroundPlane",
            "/curobo",
            "/World/mount",
        ],
    ).get_collision_check_world()

    ik_solver.update_world(obstacles)

    return ik_solver


def initialize_simulation(mesh_path: str, robot_config_file: str) -> WorldState:
    """Initialize Isaac Sim world and all components"""
    print_section_header("INITIALIZING SIMULATION", width=70)
    print()

    my_world = create_world()
    robot_state = setup_robot(my_world, robot_config_file)

    usd_helper = UsdHelper()
    target_object_prim = setup_object_from_mesh(my_world, mesh_path, usd_helper)

    camera = setup_camera(robot_state['robot_prim_path'], my_world)
    ik_solver = setup_collision_checker(my_world, robot_state, mesh_path)

    return WorldState(
        world=my_world,
        target_object_prim=target_object_prim,
        robot=robot_state['robot'],
        idx_list=robot_state['idx_list'],
        ik_solver=ik_solver,
    )


# ============================================================================
# Section 5: Simulation Execution
# ============================================================================

def get_active_joint_positions(robot, idx_list: List[int]) -> np.ndarray:
    """Get current joint positions for active joints"""
    all_positions = robot.get_joint_positions()
    return np.asarray([all_positions[i] for i in idx_list], dtype=np.float64)


def run_simulation(
    world_state: WorldState,
    joint_targets: List[np.ndarray],
    visualize_spheres: bool = False
):
    """
    Run Isaac Sim simulation with planned trajectory

    Direct waypoint execution - NO interpolation (trajectory pre-computed in Step 2)

    Args:
        world_state: WorldState containing world, robot, and IK solver
        joint_targets: List of joint configurations to execute
        visualize_spheres: Visualize robot collision spheres (default: False)
    """
    print_section_header("STARTING SIMULATION", width=70)
    print_key_value("Total waypoints", len(joint_targets))
    print_key_value("Execution mode", "Direct (no interpolation)")
    print("  Trajectory contains pre-computed collision-free configurations")
    print("  Waypoints will be executed directly without additional interpolation")
    if joint_targets:
        print("  Last joint will stay fixed to its initial value")
    print()

    # Setup trajectory queue
    target_queue: Deque[np.ndarray] = deque(joint_targets)
    fixed_last_joint_value = joint_targets[0][-1] if joint_targets else None

    step_counter = 0
    idle_counter = 0
    waypoint_counter = 0

    # Time tracking
    start_time = None
    end_time = None

    # Sphere visualization
    spheres = None
    tensor_args = TensorDeviceType()

    # Main simulation loop
    while simulation_app.is_running():
        world_state.world.step(render=True)

        if not world_state.world.is_playing():
            if idle_counter % 100 == 0:
                print("**** Click Play to start simulation *****")
            idle_counter += 1
            continue

        # Start timer when simulation actually begins
        if start_time is None:
            start_time = time.time()
            print(f"Simulation started at {time.strftime('%H:%M:%S')}")

        idle_counter = 0
        step_counter += 1

        # Visualize robot spheres
        if visualize_spheres and step_counter % 2 == 0:
            # Get current joint state from simulator
            sim_js = world_state.robot.get_joints_state()
            sim_js_names = world_state.robot.dof_names

            # Convert to CuRobo joint state
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(world_state.ik_solver.kinematics.joint_names)

            # Get sphere representation
            sph_list = world_state.ik_solver.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # Create spheres
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                # Update sphere positions and radii
                for si, s in enumerate(sph_list[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))

        # Execute trajectory waypoints directly
        # No interpolation - trajectory already contains all collision-checked configurations
        if target_queue:
            # Get next waypoint and execute directly
            next_waypoint = target_queue.popleft()

            # Keep last joint locked to avoid rotating the tool during playback
            if fixed_last_joint_value is not None and next_waypoint.size > 0:
                next_waypoint = np.copy(next_waypoint)
                next_waypoint[-1] = fixed_last_joint_value

            world_state.robot.set_joint_positions(next_waypoint.tolist(), world_state.idx_list)
            waypoint_counter += 1

        # Check if trajectory complete
        if not target_queue:
            end_time = time.time()
            elapsed_time = end_time - start_time

            print_section_header("SIMULATION COMPLETED", width=70)
            print_key_value("Waypoints executed", waypoint_counter)
            print_key_value("Total execution time", f"{elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
            print_key_value("Average time per waypoint", f"{elapsed_time/waypoint_counter:.4f}s")
            print_key_value("Waypoint execution rate", f"{waypoint_counter/elapsed_time:.2f} waypoints/s")
            print()
            break


# ============================================================================
# Section 6: Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    # Determine trajectory filename
    if args.trajectory_file is not None:
        # Custom filename specified
        trajectory_filename = args.trajectory_file
    elif args.tilt:
        # Tilt mode enabled
        trajectory_filename = "tilt_trajectory.csv"
    else:
        # Default trajectory
        trajectory_filename = "trajectory.csv"

    # Resolve paths using auto-path generation
    trajectory_path = str(config.get_trajectory_path(args.object, args.num_viewpoints, trajectory_filename))
    mesh_path = str(config.get_mesh_path(args.object, mesh_type="source"))

    print_section_header("SIMULATE TRAJECTORY", width=70)

    # Show configuration
    print_key_value("Object", args.object)
    print_key_value("Num viewpoints", args.num_viewpoints)
    print_key_value("Trajectory type", "Tilt" if args.tilt else "Regular")
    print_key_value("Trajectory file", trajectory_filename)
    print_key_value("Trajectory path", trajectory_path)
    print_key_value("Mesh path", mesh_path)
    print_key_value("Robot config", args.robot)
    print_key_value("Visualize spheres", "Enabled" if args.visualize_spheres else "Disabled")
    print_key_value("Debug mode", "Enabled" if args.debug else "Disabled")
    if args.debug:
        print("  → Target waypoint positions will be visualized as green points")
    print()

    # Validate input exists
    if not os.path.exists(trajectory_path):
        print_error(f"Trajectory file not found: {trajectory_path}")
        return 1

    # Load joint trajectory
    print_section_header("LOADING JOINT TRAJECTORY", width=70)
    print_key_value("Input file", trajectory_path)

    # Use inlined utility to load trajectory
    trajectory, joint_names = load_trajectory_csv(trajectory_path, joint_prefix="")

    # Convert to list of arrays
    joint_targets = [np.array(cfg, dtype=np.float64) for cfg in trajectory]

    print_key_value("Loaded waypoints", len(joint_targets))
    print()

    # Initialize simulation
    world_state = initialize_simulation(mesh_path, args.robot)

    # Enable extensions if headless
    if args.headless is not None:
        add_extensions(simulation_app, args.headless)

    # Run simulation
    run_simulation(world_state, joint_targets, visualize_spheres=args.visualize_spheres)

    simulation_app.close()


if __name__ == "__main__":
    main()
