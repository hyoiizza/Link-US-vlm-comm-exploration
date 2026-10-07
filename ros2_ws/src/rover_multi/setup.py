from glob import glob

from setuptools import setup

package_name = 'rover_multi'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyo',
    maintainer_email='songhyojung0807@gmail.com',
    description='Multi-robot bringup (namespaces, TF prefixes) and shared world frame for the rovers',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'world_frame_publisher = rover_multi.world_frame_publisher:main',
            'team_coordinator = rover_multi.team_coordinator:main',
            'team_agent = rover_multi.team_agent:main',
            'wifi_monitor = rover_multi.wifi_monitor:main',
            'depth_obstacle_scan = rover_multi.depth_obstacle_scan:main',
            'link_probe = rover_multi.link_probe:main',
            'peer_obstacle_scan = rover_multi.peer_obstacle_scan:main',
        ],
    },
)
