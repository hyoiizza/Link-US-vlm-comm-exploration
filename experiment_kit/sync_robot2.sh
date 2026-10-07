#!/usr/bin/env bash
# Push robot1's source changes to robot2 and rebuild there (robot2 is not a symlink install).
# Keeps robot2-only files: its ekf.yaml (no IMU), package.xml edits, engines, patch32 json.
source ~/experiment_kit/config.sh
$SSH true || { echo "robot2 ($ROBOT2_HOST) not reachable"; exit 1; }
cd ~/ros2_ws/src
rsync -a --itemize-changes --exclude='*.engine' --exclude='siglip2-base-patch32-256' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='/ros2_rover/build' --exclude='/ros2_rover/install' --exclude='/ros2_rover/log' \
  --exclude='.vscode' --exclude='*.pyc' --exclude='.git' \
  --exclude='/rf2o_laser_odometry/package.xml' --exclude='/ros2_rover/rover_description/package.xml' \
  --exclude='/ros2_rover/rover_localization/config/ekf.yaml' --exclude='/rover_vlm/models/siglip2_base_patch32_256.json' \
  ./ ${ROBOT2_USER}@${ROBOT2_HOST}:ros2_ws/src/ | grep '^<f' | awk '{print "  sent", $2}'
rsync -a ~/experiment_kit/ ${ROBOT2_USER}@${ROBOT2_HOST}:experiment_kit/
rsync -a ~/robot2_setup/ ${ROBOT2_USER}@${ROBOT2_HOST}:robot2_setup/
$SSH 'cd ~/ros2_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select rover_multi rover_vlm rover_object rover_motor_controller_cpp rover_navigation rover_localization 2>&1 | grep -E "Finished|Failed|Summary"'
