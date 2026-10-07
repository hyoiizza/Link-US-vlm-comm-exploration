"""RViz for the team: both robots in the shared world frame + the coordinator's goal selection.

    ros2 launch rover_multi team_rviz.launch.py

Run it on the laptop or on a robot with a screen. The world frame comes from
world_frame_publisher (use_coordinator:=true on the leader robot).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz = os.path.join(get_package_share_directory('rover_multi'), 'rviz', 'team.rviz')
    return LaunchDescription([
        Node(package='rviz2', executable='rviz2', name='team_rviz', arguments=['-d', rviz], output='log'),
    ])
