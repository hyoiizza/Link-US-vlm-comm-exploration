import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('rover_object'), 'config', 'object_detector.yaml')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use /clock (Gazebo or rosbag play --clock)'),
        DeclareLaunchArgument('use_mapper', default_value='true',
                              description='Project detections into the map frame'),
        Node(
            package='rover_object',
            executable='object_detector',
            name='object_detector',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rover_object',
            executable='object_mapper',
            name='object_mapper',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('use_mapper')),
        ),
    ])
