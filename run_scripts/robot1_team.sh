#!/usr/bin/env bash
# robot1: coordinator + exploration (method $1, default frontier). Ctrl+C here stops robot1.
source ~/run_scripts/env.sh
M=${1:-frontier}
ros2 launch rover_multi robot.launch.py robot_name:=robot1 use_navigation:=true use_exploration:=true \
  exploration_mode:=team use_coordinator:=true selection_method:=$M use_vlm:=false \
  motor_controller_device:=/dev/lx16a lidar_port:=/dev/rplidar \
  database_path:=$HOME/maps/$(date +%m%d_%H%M%S)_robot1_team_$M.db
