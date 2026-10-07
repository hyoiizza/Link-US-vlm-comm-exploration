"""Publish the shared world frame: world -> <robot>/map for every robot.

Run once for the team (base station or one of the robots):

    ros2 launch rover_multi world.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('rover_multi'), 'config', 'world.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(
            package='rover_multi',
            executable='world_frame_publisher',
            name='world_frame_publisher',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
