from glob import glob

from setuptools import setup

package_name = 'rover_vlm'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        # Engine + the text embeddings / preprocessing it was exported with.
        ('share/' + package_name + '/models', glob('models/*.engine') + glob('models/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hyo',
    maintainer_email='songhyojung0807@gmail.com',
    description='VLM (SigLIP 2) search-task relevance of frontiers with TensorRT',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vlm_relevance = rover_vlm.vlm_relevance_node:main',
        ],
    },
)
