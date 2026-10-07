#!/usr/bin/env bash
# Both robots' topics, recorded on robot1. Stop with Ctrl+C BEFORE stopping the robots.
source ~/run_scripts/env.sh
sleep 10
ros2 bag record -o ~/bags/$(date +%Y%m%d_%H%M%S)_team_${1:-test} \
  -e '^/(team/.*|tf|tf_static|rosout|robot[12]/(explore/.*|map|odom|plan|cmd_vel))$'
