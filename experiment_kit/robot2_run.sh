#!/usr/bin/env bash
# Runs ON robot2 (started by start_pair.sh over ssh). Every background job gets </dev/null: an
# inherited ssh stdin kept the ssh call open until the run ended (2026-10-02: start_pair hung there,
# was interrupted, and robot1 recorded nothing). Args: run_name use_vlm(0/1) robot1_ip [dry 0/1]
RUN=$1; VLM=$2; PEER=$3; DRY=${4:-0}
[ "$DRY" = 1 ] && SERVOS=1,2,3,4,5,6,7,8,9,10 || SERVOS=6     # dry run: no servo moves at all
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash
D=~/experiment_runs/$RUN; mkdir -p $D
[ "$VLM" = 1 ] && V=true || V=false
setsid nohup ros2 launch rover_multi robot.launch.py robot_name:=robot2 use_imu:=false disabled_servo_ids:=$SERVOS \
  use_navigation:=true use_exploration:=true exploration_mode:=team use_coordinator:=false use_vlm:=$V use_object_detection:=true \
  motor_controller_device:=/dev/lx16a lidar_port:=/dev/rplidar database_path:=$D/robot2.db > $D/robot2_launch.log 2>&1 < /dev/null &
echo $! > $D/robot2_launch.pid
setsid nohup ros2 run rover_multi link_probe --ros-args -r __ns:=/robot2 > $D/robot2_probe_ap.log 2>&1 < /dev/null &
echo $! > $D/robot2_probe_ap.pid
setsid nohup ros2 run rover_multi link_probe --ros-args -r __ns:=/robot2 -r __node:=link_probe_peer \
  -p target:=$PEER -p topic:=explore/link_probe_peer > $D/robot2_probe_peer.log 2>&1 < /dev/null &
echo $! > $D/robot2_probe_peer.pid
sleep 6
# robot2's own record: what it sent (team topics), its map and trajectory. Local, no Wi-Fi load.
setsid nohup ros2 bag record -o $D/robot2_bag \
  -e '^/(team/.*|tf|tf_static|rosout|robot2/(explore/.*|map|odom|plan|cmd_vel|depth_scan|object_detector/detections|object_mapper/(detections_3d|objects)))$' > $D/robot2_bag.log 2>&1 < /dev/null &
echo $! > $D/robot2_bag.pid
echo "robot2 started $RUN (vlm $V)"
