# SPDX-License-Identifier: Apache-2.0
#
# UR20 + MoveIt(cuMotion 4.4) ↔ Isaac Sim, STATE-SYNCED startup.
#
# Derived from isaac_ros_cumotion_examples/launch/ur_isaac_sim.launch.py (4.4)
# but adds the gating from the (older) ur_isaac_sim_state_synced variant so that:
#   - At Isaac "Play", MoveIt/RViz seed their state FROM the Isaac robot's actual
#     joints (/isaac_joint_states), instead of snapping the Isaac robot to MoveIt's
#     default pose.
#   - The TopicBasedSystem publishes commands to /isaac_joint_commands_raw; a relay
#     forwards raw -> /isaac_joint_commands ONLY while a trajectory goal is actively
#     executing AND Isaac state is fresh. So idle MoveIt never drives the robot;
#     only Plan&Execute does.
#
# All custom files live in this project (scripts/moveit/), so they survive
# container recreation. cuMotion/MoveIt/UR packages come from the jazzy apt install.
#
# Run (inside ros-jazzy container, system ROS sourced):
#   ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py
#   (ur_type defaults to ur20)

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import xacro
import yaml

# This project's scripts/moveit directory (where the gated xacros, xrdf, relay
# and gate scripts live).
MOVEIT_DIR = os.path.dirname(os.path.abspath(__file__))
GATED_URDF_XACRO = os.path.join(MOVEIT_DIR, 'ur_config', 'ur_gated.urdf.xacro')
ROS2_CONTROLLERS = os.path.join(MOVEIT_DIR, 'ur_config', 'ros2_controllers.yaml')
RELAY_SCRIPT = os.path.join(MOVEIT_DIR, 'isaac_joint_command_relay.py')
GATE_SCRIPT = os.path.join(MOVEIT_DIR, 'wait_for_joint_state_gate.py')

# MoveIt's default controller for UR is scaled_joint_trajectory_controller
# (ur_moveit_config moveit_controllers.yaml: default=true). The relay watches its
# action status to know when execution is active.
ACTIVE_CONTROLLER = 'scaled_joint_trajectory_controller'
STATUS_TOPIC = f'/{ACTIVE_CONTROLLER}/follow_joint_trajectory/_action/status'


def get_robot_description_contents(ur_type: str, output_file: str) -> str:
    """Process the gated UR urdf.xacro to URDF xml and dump it to output_file."""
    xacro_processed = xacro.process_file(
        GATED_URDF_XACRO,
        mappings={'ur_type': ur_type, 'name': f'{ur_type}_robot'},
    )
    robot_description = xacro_processed.toxml()
    with open(output_file, 'w') as f:
        f.write(robot_description)
    return robot_description


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[object]:
    del args, kwargs
    ur_type = str(context.perform_substitution(LaunchConfiguration('ur_type')))
    xrdf_path = str(context.perform_substitution(LaunchConfiguration('xrdf_path')))

    moveit_config = (
        MoveItConfigsBuilder(ur_type, package_name='ur_moveit_config')
        .robot_description(file_path=GATED_URDF_XACRO, mappings={'ur_type': ur_type})
        .robot_description_semantic(file_path='srdf/ur.srdf.xacro', mappings={'name': 'ur'})
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .joint_limits(file_path='config/joint_limits.yaml')
        .to_moveit_configs()
    )

    # Add cuMotion as the (default) planning pipeline.
    cumotion_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config', 'isaac_ros_cumotion_planning.yaml',
    )
    with open(cumotion_config_file_path) as f:
        cumotion_config = yaml.safe_load(f)
    moveit_config.planning_pipelines['planning_pipelines'].insert(0, 'isaac_ros_cumotion')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'
    moveit_config.moveit_cpp.update({'use_sim_time': True})
    # Tolerate small start-state mismatch (MoveIt seed vs Isaac state).
    moveit_config.trajectory_execution['trajectory_execution']['allowed_start_tolerance'] = 0.1

    urdf_path = '/tmp/collated_ur20_urdf.urdf'
    robot_description = get_robot_description_contents(ur_type, urdf_path)

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        remappings=[('/joint_states', '/isaac_joint_states')],
    )

    world2robot_tf_node = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='static_transform_publisher', output='log',
        arguments=['--frame-id', 'world', '--child-frame-id', 'base_link'],
        parameters=[{'use_sim_time': True}],
    )

    ros2_control_node = Node(
        package='controller_manager', executable='ros2_control_node',
        parameters=[ROS2_CONTROLLERS, {'use_sim_time': True}],
        remappings=[('/controller_manager/robot_description', '/robot_description')],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    scaled_jtc_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=[ACTIVE_CONTROLLER, '-c', '/controller_manager'],
    )
    # joint_trajectory_controller is the INSPECTION controller (Publish targets it).
    # Loaded INACTIVE; the Isaac UI activates it (and deactivates scaled) when the
    # pipeline mode is Inspection, so MoveIt (which uses scaled) is blocked then.
    inspection_jtc_spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['joint_trajectory_controller', '-c', '/controller_manager', '--inactive'],
    )

    move_group_node = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[moveit_config.to_dict()],
        arguments=['--ros-args', '--log-level', 'info'],
    )
    rviz_config_file = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'),
        'rviz', 'ur_moveit_config.rviz',
    )
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': True},
        ],
    )
    # cuMotion 4.4 backend: the MoveIt plugin (loaded in move_group via the pipeline
    # config above) forwards planning to the cuMotion ACTION SERVER, which runs as
    # composable nodes (CumotionPlanner + StaticPlanningSceneServer) brought up by
    # the package's own launch file. We include it and override args for UR20.
    #   - urdf/xrdf: our dumped UR20 urdf + project ur20.xrdf
    #   - read_esdf_world/update_esdf_on_request = False: no nvblox here, so don't
    #     block planning waiting on an ESDF service.
    #   - joint_states_topic /joint_states: seed planner from the broadcaster, which
    #     mirrors Isaac's actual state.
    cumotion_launch = os.path.join(
        get_package_share_directory('isaac_ros_cumotion'),
        'launch', 'isaac_ros_cumotion.launch.py',
    )
    cumotion_action_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cumotion_launch),
        launch_arguments={
            'cumotion_action_server.urdf_file_path': urdf_path,
            'cumotion_action_server.xrdf_file_path': xrdf_path,
            'cumotion_action_server.read_esdf_world': 'False',
            'cumotion_action_server.update_esdf_on_request': 'False',
            'cumotion_action_server.joint_states_topic': '/joint_states',
        }.items(),
    )

    # Relay: forward /isaac_joint_commands_raw -> /isaac_joint_commands only while a
    # trajectory goal is active on EITHER controller (scaled = MoveIt, jtc =
    # Inspection). No --status-topic → relay watches both by default. This keeps the
    # robot still when idle (MoveIt reflects the robot) and moves it only on Execute.
    isaac_joint_command_relay = ExecuteProcess(
        cmd=['python3', RELAY_SCRIPT], output='screen',
    )

    # Gates: block startup until joint states are actually flowing.
    isaac_joint_state_gate = ExecuteProcess(
        cmd=['python3', GATE_SCRIPT, '--topic', '/isaac_joint_states', '--timeout', '60.0'],
        output='screen',
    )
    ros_joint_state_gate = ExecuteProcess(
        cmd=['python3', GATE_SCRIPT, '--topic', '/joint_states', '--timeout', '60.0'],
        output='screen',
    )

    # Sequence: Isaac state present -> ros2_control + broadcaster -> /joint_states
    # present -> activate trajectory controller -> start MoveIt/RViz/cuMotion.
    start_control_after_isaac = RegisterEventHandler(OnProcessExit(
        target_action=isaac_joint_state_gate,
        on_exit=[ros2_control_node, joint_state_broadcaster_spawner, ros_joint_state_gate],
    ))
    activate_controller = RegisterEventHandler(OnProcessExit(
        target_action=ros_joint_state_gate,
        on_exit=[scaled_jtc_spawner, inspection_jtc_spawner],
    ))
    start_moveit_after_controller = RegisterEventHandler(OnProcessExit(
        target_action=scaled_jtc_spawner,
        on_exit=[move_group_node, rviz_node, cumotion_action_server],
    ))

    return [
        robot_state_publisher,
        world2robot_tf_node,
        isaac_joint_command_relay,
        isaac_joint_state_gate,
        start_control_after_isaac,
        activate_controller,
        start_moveit_after_controller,
    ]


def generate_launch_description():
    default_xrdf = os.path.join(MOVEIT_DIR, 'ur20.xrdf')
    launch_args = [
        DeclareLaunchArgument(
            'ur_type', default_value='ur20',
            description='UR robot type',
            choices=['ur3', 'ur3e', 'ur5', 'ur5e', 'ur10', 'ur10e', 'ur16e', 'ur20', 'ur30'],
        ),
        DeclareLaunchArgument(
            'xrdf_path', default_value=default_xrdf,
            description='Absolute path to the cuMotion XRDF for this robot.',
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
