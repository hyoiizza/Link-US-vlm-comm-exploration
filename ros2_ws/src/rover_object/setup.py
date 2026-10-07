from glob import glob

from setuptools import setup

package_name = 'rover_object'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.onnx')),
        ('share/' + package_name + '/models/optimized',
         glob('models/optimized/*.engine') + glob('models/optimized/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyo',
    maintainer_email='songhyojung0807@gmail.com',
    description='YOLO (person/fire) object detection for the rover, accelerated with TensorRT',
    license='AGPL-3.0',
    entry_points={
        'console_scripts': [
            'object_detector = rover_object.detector_node:main',
            'object_mapper = rover_object.object_mapper_node:main',
        ],
    },
)
