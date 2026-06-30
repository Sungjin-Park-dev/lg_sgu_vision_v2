# SPDX-License-Identifier: Apache-2.0
#
# UR20 REAL robot + MoveIt(cuMotion) — Run mode = real of the 2-axis model.
#
# Brings up the real robot stack (ur_robot_driver; default mock_hardware) and a
# move_group with the cuMotion planning pipeline, executing on the REAL robot's
# standard scaled_joint_trajectory_controller. The Isaac app (run mode = real)
# mirrors the real /joint_states as a digital twin — it is NOT driven by this
# stack. publish_trajectory.py also targets /scaled_joint_trajectory_controller,
# so in real mode both MoveIt Execute and Inspection Publish drive the real robot.
#
# Unlike the sim stack (ur20_isaac_state_synced.launch.py): no TopicBasedSystem,
# no relay/gate/clock — the real driver always publishes /joint_states and the
# stack runs on real (system) time.
#
# Run (inside ros-jazzy, system ROS sourced; cuMotion overlay not needed):
#   ros2 launch scripts/moveit/ur20_real_moveit.launch.py            # mock by default
#   ros2 launch scripts/moveit/ur20_real_moveit.launch.py use_mock_hardware:=false robot_ip:=<ip>

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
import xacro
import yaml

MOVEIT_DIR = os.path.dirname(os.path.abspath(__file__))
# robot_description for move_group / cuMotion (ur_description ur20 kinematics; the
# ros2_control tag inside is irrelevant here — execution goes via the controller
# action to the driver, not through this description).
UR_URDF_XACRO = os.path.join(
    get_package_share_directory('isaac_ros_cumotion_examples'), 'ur_config', 'ur.urdf.xacro')


def launch_setup(context: LaunchContext, *args, **kwargs):
    del args, kwargs
    ur_type = str(context.perform_substitution(LaunchConfiguration('ur_type')))
    xrdf_path = str(context.perform_substitution(LaunchConfiguration('xrdf_path')))

    # 1) Real robot driver: ros2_control + scaled_joint_trajectory_controller + /joint_states.
    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ur_robot_driver'), 'launch', 'ur_control.launch.py')),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': LaunchConfiguration('robot_ip'),
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            'launch_rviz': 'false',
        }.items(),
    )

    # 2) move_group with cuMotion pipeline (real time).
    moveit_config = (
        MoveItConfigsBuilder(ur_type, package_name='ur_moveit_config')
        .robot_description(file_path=UR_URDF_XACRO, mappings={'ur_type': ur_type})
        .robot_description_semantic(file_path='srdf/ur.srdf.xacro', mappings={'name': 'ur'})
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .joint_limits(file_path='config/joint_limits.yaml')
        .to_moveit_configs()
    )
    cumotion_cfg = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_moveit'),
        'config', 'isaac_ros_cumotion_planning.yaml')
    with open(cumotion_cfg) as f:
        cm = yaml.safe_load(f)
    moveit_config.planning_pipelines['planning_pipelines'].insert(0, 'isaac_ros_cumotion')
    moveit_config.planning_pipelines['isaac_ros_cumotion'] = cm
    moveit_config.planning_pipelines['default_planning_pipeline'] = 'isaac_ros_cumotion'
    moveit_config.moveit_cpp.update({'use_sim_time': False})
    moveit_config.trajectory_execution['trajectory_execution']['allowed_start_tolerance'] = 0.1

    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[moveit_config.to_dict(), {'use_sim_time': False}],
        arguments=['--ros-args', '--log-level', 'info'],
    )
    rviz_cfg = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'), 'rviz', 'ur_moveit_config.rviz')
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', rviz_cfg],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': False},
        ],
    )

    # 3) cuMotion action server (needs the urdf as a file).
    urdf_path = '/tmp/collated_ur20_real_urdf.urdf'
    with open(urdf_path, 'w') as f:
        f.write(xacro.process_file(
            UR_URDF_XACRO, mappings={'ur_type': ur_type, 'name': f'{ur_type}_robot'}).toxml())
    cumotion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('isaac_ros_cumotion'),
            'launch', 'isaac_ros_cumotion.launch.py')),
        launch_arguments={
            'cumotion_action_server.urdf_file_path': urdf_path,
            'cumotion_action_server.xrdf_file_path': xrdf_path,
            'cumotion_action_server.read_esdf_world': 'False',
            'cumotion_action_server.update_esdf_on_request': 'False',
            'cumotion_action_server.joint_states_topic': '/joint_states',
        }.items(),
    )

    # Delay move_group/RViz/cuMotion so the driver's controllers are up first.
    delayed = TimerAction(period=6.0, actions=[move_group, rviz, cumotion])

    # cuMotion trajectories end with a tiny non-zero velocity; UR's scaled (and
    # plain) joint_trajectory_controller reject a goal whose last point has nonzero
    # velocity unless this flag is set, so MoveIt Execute would abort with
    # "Velocity of last trajectory point ... is not zero". Set it at runtime once the
    # controllers are up (verified to take effect on the running controller). Retry a
    # few times in case the controller spawns late.
    allow_nonzero_vel = TimerAction(period=8.0, actions=[ExecuteProcess(
        cmd=['bash', '-c',
             'for i in 1 2 3 4 5 6; do '
             'ros2 param set /scaled_joint_trajectory_controller '
             'allow_nonzero_velocity_at_trajectory_end true && '
             'ros2 param set /joint_trajectory_controller '
             'allow_nonzero_velocity_at_trajectory_end true && break; '
             'sleep 2; done'],
        output='screen')])
    return [ur_control, delayed, allow_nonzero_vel]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument('ur_type', default_value='ur20'),
        DeclareLaunchArgument('robot_ip', default_value='0.0.0.0',
                              description='Robot IP (ignored for mock_hardware).'),
        DeclareLaunchArgument('use_mock_hardware', default_value='true',
                              description='true = ur_robot_driver mock (no real robot).'),
        DeclareLaunchArgument('xrdf_path', default_value=os.path.join(MOVEIT_DIR, 'ur20.xrdf')),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
