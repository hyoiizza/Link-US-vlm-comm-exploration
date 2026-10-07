"""Bring up one rover of the team under its own namespace and TF prefix.

    ros2 launch rover_multi robot.launch.py robot_name:=robot1

Everything runs under /<robot_name>/..., and every TF frame is prefixed with
"<robot_name>/" so all robots can share the global /tf tree under ``world``
(see world.launch.py). The single-robot launch files in ros2_rover are left
untouched; their parameter files are reused and rewritten on the fly.
"""
import importlib.util
import copy
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node, PushRosNamespace
from launch_ros.descriptions import ComposableNode

from rover_multi.namespacing import prefix_frame, rewrite_params_file


def _truthy(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ('true', '1', 'yes')


def orbbec_camera(config_file, extra_params):
    """Orbbec Gemini 330-series driver with extra node parameters.

    gemini_330_series.launch.py only takes yaml keys that are also its launch
    arguments, so the per-robot frame ids cannot be passed through it. Its own
    parameter logic (argument defaults + yaml merge + type conversion) is reused
    here and the frame ids are added on top; the node is set up exactly like it does.
    """
    path = os.path.join(get_package_share_directory('orbbec_camera'), 'launch', 'gemini_330_series.launch.py')
    spec = importlib.util.spec_from_file_location('orbbec_gemini_330_series_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = [e for e in module.generate_launch_description().entities if isinstance(e, DeclareLaunchArgument)]

    ctx = LaunchContext()   # private context, so the driver's arguments do not leak into ours
    for arg in args:
        ctx.launch_configurations[arg.name] = ''.join(s.perform(ctx) for s in arg.default_value)
    ctx.launch_configurations['camera_name'] = 'camera'
    ctx.launch_configurations['config_file_path'] = config_file
    params = module.load_parameters(ctx, args)
    params.update(extra_params)

    return GroupAction([
        PushRosNamespace('camera'),
        ComposableNodeContainer(
            name='camera_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[ComposableNode(
                package='orbbec_camera',
                plugin='orbbec_camera::OBCameraNodeDriver',
                name='camera',
                parameters=[params],
            )],
            output='log',
        ),
    ])


def launch_setup(context):
    robot = LaunchConfiguration('robot_name').perform(context).strip('/')
    if not robot:
        raise RuntimeError('robot_name must not be empty (e.g. robot_name:=robot1)')
    slam = LaunchConfiguration('slam').perform(context)
    sim = {'use_sim_time': _truthy(context, 'use_sim_time')}

    bringup_dir = get_package_share_directory('rover_bringup')
    loc_dir = get_package_share_directory('rover_localization')
    desc_dir = get_package_share_directory('rover_description')

    def cfg(pkg_dir, name, overrides=None):
        return rewrite_params_file(os.path.join(pkg_dir, 'config', name), robot, overrides)

    def frame(name):
        return prefix_frame(robot, name)

    actions = [PushRosNamespace(robot)]

    # --- robot model: base_link -> sensor frames ------------------------
    if _truthy(context, 'use_robot_state_publisher'):
        doc = xacro.process_file(os.path.join(desc_dir, 'robots', 'rover.urdf.xacro'))
        actions.append(Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': doc.toxml(), 'frame_prefix': robot + '/'}, sim],
        ))

    # --- sensors & actuators --------------------------------------------
    actions.append(Node(
        package='sllidar_ros2',
        executable='sllidar_node',
        name='sllidar_node',
        output='screen',
        parameters=[cfg(bringup_dir, 'rplidar_c1.yaml', {'sllidar_node': {'ros__parameters': {
            'serial_port': LaunchConfiguration('lidar_port').perform(context)}}}), sim],
    ))
    # The lidar sees the rover's own chassis (~34% of its points); everything downstream
    # (rf2o, SLAM, costmaps) reads <robot>/scan_filtered with those returns removed.
    actions.append(Node(
        package='rover_localization',
        executable='scan_self_filter.py',
        name='scan_self_filter',
        output='log',
        parameters=[{'base_frame': frame('base_link')}, sim],
    ))

    if _truthy(context, 'use_imu'):
        actions.append(Node(
            package='bno08x_driver',
            executable='bno08x_driver',
            name='bno08x_driver',
            output='screen',
            parameters=[cfg(bringup_dir, 'bno085.yaml'), sim],
            # The driver publishes fully qualified names; pull them into the namespace.
            remappings=[('/imu', 'imu'), ('/magnetic_field', 'magnetic_field')],
        ))

    if _truthy(context, 'use_camera'):
        # The Orbbec node names frames "<camera_name>_<stream>_..."; override them with prefixed ids.
        camera_frames = {
            'camera_color_frame_id': frame('camera_color_frame'),
            'color_optical_frame_id': frame('camera_color_optical_frame'),
            'camera_depth_frame_id': frame('camera_depth_frame'),
            'depth_optical_frame_id': frame('camera_depth_optical_frame'),
        }
        actions.append(orbbec_camera(cfg(bringup_dir, 'gemini_335.yaml'), camera_frames))
        if _truthy(context, 'use_depth_obstacles'):
            # The lidar scans ~0.46 m above the floor: low obstacles only show up in the depth image.
            actions.append(Node(
                package='rover_multi',
                executable='depth_obstacle_scan',
                name='depth_obstacle_scan',
                output='log',
                parameters=[{'base_frame': frame('base_link')}, sim],
            ))

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rover_motor_controller_cpp'), 'launch', 'motor_controller.launch.py')),
        launch_arguments={
            'motor_controller_device': LaunchConfiguration('motor_controller_device').perform(context),
            'baud_rate': LaunchConfiguration('baud_rate').perform(context),
            'disabled_servo_ids': LaunchConfiguration('disabled_servo_ids').perform(context),
        }.items(),
    ))

    # --- odometry: rf2o (scan matching) + EKF -> <robot>/odom -> <robot>/base_link
    actions.append(Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        output='log',
        parameters=[cfg(loc_dir, 'rf2o.yaml', {'rf2o_laser_odometry': {'ros__parameters': {
            'laser_scan_topic': f'/{robot}/scan_filtered'}}}), sim],
    ))
    # rf2o drifts when people walk past a parked rover; hold its pose unless the rover is
    # commanded to move or the gyro sees it turning. The EKF reads the gated odometry.
    actions.append(Node(
        package='rover_localization',
        executable='odom_stationary_gate.py',
        name='odom_stationary_gate',
        output='log',
        parameters=[sim],
    ))
    actions.append(Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='log',
        parameters=[cfg(loc_dir, 'ekf.yaml'), sim],
        remappings=[('odometry/filtered', 'odom'), ('accel/filtered', 'accel'),
                    ('odom_rf2o', 'odom_rf2o_gated')],
    ))

    # --- SLAM: <robot>/map -> <robot>/odom, occupancy grid on /<robot>/map
    if slam == 'rtabmap':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(loc_dir, 'launch', 'rtabmap.launch.py')),
            launch_arguments={
                'use_sim_time': str(sim['use_sim_time']),
                'base_frame': frame('base_link'),
                'map_frame': frame('map'),
                'odom_frame': frame('odom'),
                'map_topic': 'map',
                'scan_topic': 'scan_filtered',
                'imu_topic': 'imu',
                'database_path': os.path.expanduser(
                    LaunchConfiguration('database_path').perform(context)
                    or f'~/.ros/rtabmap_{robot}.db'),
                'delete_db': LaunchConfiguration('delete_db').perform(context),
                'launch_rtabmapviz': LaunchConfiguration('launch_rtabmapviz').perform(context),
            }.items(),
        ))
    elif slam == 'slam_toolbox':
        actions.append(Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[cfg(loc_dir, 'slam_toolbox_mapping.yaml', {
                'slam_toolbox': {'ros__parameters': {'map_name': f'/{robot}/map',
                                                     'scan_topic': f'/{robot}/scan_filtered'}}}), sim],
        ))

    # --- person / fire detection placed in <robot>/map --------------------
    if _truthy(context, 'use_object_detection'):
        obj_dir = get_package_share_directory('rover_object')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(obj_dir, 'launch', 'object_detector.launch.py')),
            launch_arguments={
                'params_file': cfg(obj_dir, 'object_detector.yaml'),
                'use_sim_time': str(sim['use_sim_time']).lower(),
            }.items(),
        ))

    # --- floor / wall / stairs segmentation of the color camera -----------
    if _truthy(context, 'use_segmentation'):
        sem_dir = get_package_share_directory('rover_semantic')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(sem_dir, 'launch', 'semantic_segmentation.launch.py')),
            launch_arguments={
                'params_file': cfg(sem_dir, 'semantic_segmentation.yaml'),
                'use_sim_time': str(sim['use_sim_time']).lower(),
            }.items(),
        ))

    # --- Nav2: /<robot>/navigate_to_pose, commands on /<robot>/cmd_vel ------
    if _truthy(context, 'use_navigation'):
        nav_dir = get_package_share_directory('rover_navigation')
        planner = LaunchConfiguration('planner').perform(context)
        controller = LaunchConfiguration('controller').perform(context)
        map_topic = f'/{robot}/map'
        obstacle_layer = {'scan': {'topic': f'/{robot}/scan_filtered'}}
        sources = ['scan']
        if _truthy(context, 'use_camera') and _truthy(context, 'use_depth_obstacles'):
            # Low obstacles from depth_obstacle_scan (base_link frame, camera FOV only). The camera
            # sees the floor only from ~1.2 m ahead: inside raytrace_min_range nothing is cleared,
            # so an obstacle that drops into that blind zone stays in the costmap.
            sources.append('depth_scan')
            obstacle_layer['depth_scan'] = {
                'topic': f'/{robot}/depth_scan', 'data_type': 'LaserScan',
                'obstacle_max_range': 3.0, 'obstacle_min_range': 0.0,
                'raytrace_max_range': 3.0, 'raytrace_min_range': 1.3,
                'max_obstacle_height': 2.0, 'min_obstacle_height': -2.0,
                'inf_is_valid': True, 'clearing': True, 'marking': True}
        if (_truthy(context, 'use_peer_obstacles') and _truthy(context, 'use_exploration')
                and LaunchConfiguration('exploration_mode').perform(context) == 'team'):
            # The other robot from the shared /tf (peer_obstacle_scan): the lidar scans over its
            # body. Marking only; the lidar's raytracing clears the spot once the peer moves on.
            sources.append('peer_scan')
            obstacle_layer['peer_scan'] = {
                'topic': f'/{robot}/peer_scan', 'data_type': 'LaserScan',
                'obstacle_max_range': 8.0, 'obstacle_min_range': 0.0,
                'raytrace_max_range': 8.0, 'raytrace_min_range': 0.0,
                'max_obstacle_height': 2.0, 'min_obstacle_height': -2.0,
                'inf_is_valid': False, 'clearing': False, 'marking': True}
        obstacle_layer['observation_sources'] = ' '.join(sources)
        # Nav2 nests the file under the namespace itself (RewrittenYaml root_key).
        nav_params = rewrite_params_file(
            os.path.join(nav_dir, 'params', f'{planner}_{controller}.yaml'), robot,
            overrides={
                'global_costmap': {'global_costmap': {'ros__parameters': {
                    'map_topic': map_topic, 'static_layer': {'map_topic': map_topic},
                    # separate copies: one shared dict would be dumped as a YAML alias,
                    # which the ROS params parser rejects ("Will not support aliasing")
                    'obstacle_layer': copy.deepcopy(obstacle_layer)}}},
                'local_costmap': {'local_costmap': {'ros__parameters': {
                    'obstacle_layer': copy.deepcopy(obstacle_layer)}}}},
            wrap_namespace=False)
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav_dir, 'launch', 'navigation.launch.py')),
            launch_arguments={
                'namespace': robot,
                'use_global_tf': 'True',
                'use_sim_time': str(sim['use_sim_time']),
                'autostart': 'true',
                'use_composition': 'False',
                'params_file': nav_params,
                # controller/behaviors -> cmd_vel_nav -> velocity_smoother -> cmd_vel -> vel_parser
                'cmd_vel_topic': 'cmd_vel_nav',
                'default_nav_to_pose_bt_xml': os.path.join(nav_dir, 'behavior_trees', 'rover_bt.xml'),
            }.items(),
        ))

    # --- frontier exploration ------------------------------------------------
    #   solo: the explorer picks goals and drives this robot's Nav2 by itself
    #   team: the explorer only ranks frontiers; team_agent bids on them and
    #         drives the tasks the coordinator awards (standalone fallback)
    mode = LaunchConfiguration('exploration_mode').perform(context)
    team_params = os.path.join(get_package_share_directory('rover_multi'), 'config', 'team.yaml')
    if _truthy(context, 'use_exploration'):
        frontier_dir = get_package_share_directory('frontier_exploration_ros2')
        actions.append(Node(
            package='frontier_exploration_ros2',
            executable='frontier_explorer',
            name='frontier_explorer',
            output='screen',
            parameters=[cfg(frontier_dir, 'params.yaml', {'frontier_explorer': {'ros__parameters': {
                'external_assignment_enabled': mode == 'team'}}}), sim],
        ))
        if mode == 'team':
            if _truthy(context, 'use_peer_obstacles'):
                actions.append(Node(
                    package='rover_multi',
                    executable='peer_obstacle_scan',
                    name='peer_obstacle_scan',
                    output='log',
                    parameters=[{'robot_name': robot}, sim],
                ))
            actions.append(Node(
                package='rover_multi',
                executable='team_agent',
                name='team_agent',
                output='screen',
                parameters=[team_params, {'robot_name': robot}, sim],
            ))
            if _truthy(context, 'use_vlm'):
                # Search-task relevance S_j of this robot's frontiers, on this robot's Jetson.
                vlm_dir = get_package_share_directory('rover_vlm')
                actions.append(Node(
                    package='rover_vlm',
                    executable='vlm_relevance',
                    name='vlm_relevance',
                    output='log',
                    parameters=[cfg(vlm_dir, 'vlm.yaml'), sim],
                ))
            # Measured Wi-Fi for the coordinator's link model and the link metrics (every method).
            actions.append(Node(
                package='rover_multi',
                executable='wifi_monitor',
                name='wifi_monitor',
                output='log',
                parameters=[{'robot_name': robot,
                             'interface': LaunchConfiguration('wifi_interface').perform(context)}, sim],
            ))

    team_actions = []
    if _truthy(context, 'use_coordinator'):
        # Upper layer + shared world frame, once for the team (leader robot), outside the namespace.
        team_actions.append(Node(
            package='rover_multi',
            executable='world_frame_publisher',
            name='world_frame_publisher',
            output='screen',
            parameters=[os.path.join(get_package_share_directory('rover_multi'), 'config', 'world.yaml'), sim],
        ))
        coordinator_params = {'host_robot': robot}
        method = LaunchConfiguration('selection_method').perform(context)
        if method:
            coordinator_params['selection.method'] = method
        team_actions.append(Node(
            package='rover_multi',
            executable='team_coordinator',
            name='team_coordinator',
            output='screen',
            parameters=[team_params, coordinator_params, sim],
        ))

    return [GroupAction(actions)] + team_actions


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1'),
        DeclareLaunchArgument('robot_name', default_value='robot1',
                              description='Namespace and TF prefix of this robot'),
        DeclareLaunchArgument('slam', default_value='rtabmap',
                              choices=['rtabmap', 'slam_toolbox', 'off'],
                              description='Which SLAM owns <robot>/map'),
        DeclareLaunchArgument('delete_db', default_value='true',
                              description='Start RTAB-Map with an empty database'),
        DeclareLaunchArgument('database_path', default_value='',
                              description='RTAB-Map database (empty: ~/.ros/rtabmap_<robot_name>.db)'),
        DeclareLaunchArgument('launch_rtabmapviz', default_value='false'),
        DeclareLaunchArgument('disabled_servo_ids', default_value='',
                              description="LX-16A ids that get no commands (broken servo); robot2: '6'"),
        DeclareLaunchArgument('motor_controller_device', default_value='/dev/lx16a',
                              description='LX-16A bus (e.g. /dev/ttyUSB1, /dev/ttyTHS1)'),
        DeclareLaunchArgument('baud_rate', default_value='115200'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/rplidar',
                              description='RPLIDAR C1 serial port (e.g. /dev/ttyUSB0)'),
        DeclareLaunchArgument('use_robot_state_publisher', default_value='true'),
        DeclareLaunchArgument('use_imu', default_value='true'),
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_object_detection', default_value='false'),
        DeclareLaunchArgument('use_peer_obstacles', default_value='true',
                              description='Team mode: the other robots as obstacles in the costmaps '
                                          '(peer_obstacle_scan, from the shared /tf)'),
        DeclareLaunchArgument('use_depth_obstacles', default_value='true',
                              description='Low obstacles below the lidar plane from the depth camera '
                                          '(depth_obstacle_scan -> costmaps)'),
        DeclareLaunchArgument('use_segmentation', default_value='false',
                              description='Floor / wall / stairs segmentation (rover_semantic)'),
        DeclareLaunchArgument('use_navigation', default_value='false',
                              description='Run Nav2 for this robot'),
        # Smac2D: grid A* + RPP rotating in place (the rover turns on the spot). Only Smac2D_RPP
        # exists for it; SmacHybrid / SmacLattice model a car-like turning radius.
        DeclareLaunchArgument('planner', default_value='Smac2D',
                              choices=['Smac2D', 'SmacHybrid', 'SmacLattice']),
        DeclareLaunchArgument('controller', default_value='RPP', choices=['RPP', 'TEB']),
        DeclareLaunchArgument('use_exploration', default_value='false',
                              description='Run frontier exploration (needs use_navigation)'),
        DeclareLaunchArgument('exploration_mode', default_value='solo', choices=['solo', 'team'],
                              description='solo: explore alone; team: auction-based task allocation'),
        DeclareLaunchArgument('selection_method', default_value='',
                              choices=['', 'auction', 'frontier', 'comm_aware', 'semantic_only', 'proposed'],
                              description='Coordinator goal selection (empty: selection.method of team.yaml)'),
        DeclareLaunchArgument('use_vlm', default_value='false',
                              description='VLM frontier relevance on this robot (rover_vlm, needs its engine)'),
        DeclareLaunchArgument('wifi_interface', default_value='',
                              description='Wireless interface to measure (empty: first one in /proc/net/wireless)'),
        DeclareLaunchArgument('use_coordinator', default_value='false',
                              description='Leader robot: run the team coordinator and the world frame'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
