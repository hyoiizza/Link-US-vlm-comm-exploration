#!/usr/bin/env bash
# Start one experiment run on both robots from robot1.
#   bash start_pair.sh <run_name> <frontier|comm_aware|semantic_only|proposed> <use_vlm 0|1> [person_id] [note]
#                      --front F [--left L | --right R]
# --front/--left/--right: the target person, tape-measured in m from robot1's start (its centre, facing
# forward). Only YOLO persons near it count as "found" in the analysis (others are bystanders).
set -e
source ~/experiment_kit/config.sh
POS=(); TX=; TY=0
while [ $# -gt 0 ]; do
  case $1 in
    --front) TX=$2; shift 2;;
    --left)  TY=$2; shift 2;;
    --right) TY=$(python3 -c "print(-float('$2'))"); shift 2;;
    --*) echo "unknown option $1"; exit 1;;
    *) POS+=("$1"); shift;;
  esac
done
RUN=${POS[0]}; METHOD=${POS[1]}; VLM=${POS[2]}; PERSON=${POS[3]:-none}; NOTE=${POS[4]:-}
[ -z "$RUN" ] || [ -z "$METHOD" ] || [ -z "$VLM" ] && { echo "usage: start_pair.sh run method use_vlm [person] [note] --front F [--left L|--right R]"; exit 1; }
case $METHOD in frontier|comm_aware|semantic_only|proposed) ;; *) echo "!! method must be frontier|comm_aware|semantic_only|proposed (got '$METHOD')"; exit 1;; esac
case $VLM in 0|1) ;; *) echo "!! use_vlm must be 0 or 1 (got '$VLM')"; exit 1;; esac
# S needs the VLM: proposed/semantic_only without it silently become comm_aware/frontier (p1..p7, 2026-10-02)
case $METHOD in proposed|semantic_only) [ "$VLM" = 1 ] || { echo "!! $METHOD needs use_vlm 1 (e.g. start_pair.sh p_vlm_9 $METHOD 1 yolo ...)"; exit 1; };;
                 *) [ "$VLM" = 0 ] || { echo "!! $METHOD does not use S: run it with use_vlm 0"; exit 1; };; esac
[ "$PERSON" = yolo ] || { echo "!! 4th argument must be yolo (got '$PERSON'); the memo comes 5th"; exit 1; }
[ -z "$TX" ] && { echo "!! target position missing: add --front <m> and --left <m> or --right <m> (from robot1's start)"; exit 1; }
python3 -c "float('$TX'); float('$TY')" 2>/dev/null || { echo "!! --front/--left/--right need numbers (m)"; exit 1; }
python3 -c "y=float('$TY'); print(f'target: $TX m forward, {abs(y):g} m ' + ('left' if y >= 0 else 'right') + f\" of robot1's start (world x=$TX, y={y:g})\")"
D=$RUNS_DIR/$RUN; [ -e $D ] && { echo "$D exists, pick another run name"; exit 1; }
mkdir -p $D/params
source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash
R1_IP=$(ip -4 -o addr show wlP1p1s0 | awk '{print $4}' | cut -d/ -f1)
$SSH true || { echo "robot2 ($ROBOT2_HOST) not reachable"; exit 1; }
# one run at a time: a second launch fights the first for the lidar / IMU / motor ports
if pgrep -f "[r]os2 launch rover_multi robot.launch.py" >/dev/null; then
  echo "!! a robot1 launch is still running. Stop it first: bash ~/experiment_kit/stop_pair.sh <previous run>"
  echo "   (or everything: bash ~/experiment_kit/ros_cleanup.sh)"; exit 1; fi
if $SSH 'pgrep -f "[r]os2 launch rover_multi robot.launch.py" >/dev/null'; then
  echo "!! a robot2 launch is still running. Stop it: $SSH 'bash ~/experiment_kit/ros_cleanup.sh'"; exit 1; fi
# hard requirements: without them the run is wasted (2026-10-02: 3 runs without /dev/lx16a)
for dev in /dev/lx16a /dev/rplidar; do
  [ -e $dev ] || { echo "!! robot1 $dev missing. Motor board: sudo insmod ~/ch341_driver/ch341.ko"; exit 1; }
  $SSH "[ -e $dev ]" || { echo "!! robot2 $dev missing (replug USB / power)"; exit 1; }
done
OFF=$(python3 -c "import time,subprocess;t0=time.time();r=float(subprocess.check_output('$SSH date +%s.%N',shell=True));t1=time.time();print(round(r-(t0+t1)/2,3))")
if python3 -c "import sys; sys.exit(0 if abs($OFF) < 2 else 1)"; then echo "clock offset robot2-robot1: $OFF s"; else
  echo "!! robot2 clock is off by $OFF s (no RTC, no internet). Fix: bash ~/experiment_kit/set_robot2_time.sh"; exit 1; fi
echo "== preflight robot1"; bash ~/robot2_setup/preflight.sh || true
echo "== preflight robot2"; $SSH 'bash ~/robot2_setup/preflight.sh' || true
if [ "$AUTO_YES" != 1 ]; then read -p "Start run $RUN ($METHOD, vlm $VLM)? [y/N] " a; [ "$a" = y ] || exit 1; fi
cp ~/ros2_ws/src/rover_multi/config/{team.yaml,world.yaml} ~/ros2_ws/src/rover_vlm/config/vlm.yaml \
   ~/ros2_ws/src/ros2_rover/rover_navigation/params/Smac2D_RPP.yaml $D/params/
cat > $D/manifest.yaml <<M
run: $RUN
method: $METHOD
use_vlm: $VLM
person: $PERSON
target: {x: $TX, y: $TY}   # m, robot1 start frame (x forward, y left), tape-measured
note: "$NOTE"
dry_run: ${DRY:-0}
start: $(date -Iseconds)
robot1: {ip: $R1_IP, ssid: "$(iw dev wlP1p1s0 link | awk -F': ' '/SSID/ {print $2}')", power_save: $(iw dev wlP1p1s0 get power_save | awk '{print $3}')}
robot2: {ip: $ROBOT2_HOST, ssid: "$($SSH "iw dev wlP1p1s0 link | awk -F': ' '/SSID/ {print \$2}'")", power_save: $($SSH "iw dev wlP1p1s0 get power_save | awk '{print \$3}'")}
clock_offset_robot2_minus_robot1_s: $OFF
M
[ "$VLM" = 1 ] && V=true || V=false
SERVO_ARG=; [ "$DRY" = 1 ] && SERVO_ARG=disabled_servo_ids:=1,2,3,4,5,6,7,8,9,10   # DRY=1: nothing moves (kit check)
# (an empty "disabled_servo_ids:=" is rejected by ros2 launch: only pass it when set)
# robot1 first: it hosts the coordinator; robot2's agent goes standalone if no coordinator within 4 s
setsid nohup ros2 launch rover_multi robot.launch.py robot_name:=robot1 $SERVO_ARG use_navigation:=true use_exploration:=true \
  exploration_mode:=team use_coordinator:=true selection_method:=$METHOD use_vlm:=$V use_object_detection:=true \
  motor_controller_device:=/dev/lx16a lidar_port:=/dev/rplidar database_path:=$D/robot1.db > $D/robot1_launch.log 2>&1 < /dev/null &
echo $! > $D/robot1_launch.pid
setsid nohup ros2 run rover_multi link_probe --ros-args -r __ns:=/robot1 > $D/robot1_probe_ap.log 2>&1 < /dev/null &
echo $! > $D/robot1_probe_ap.pid
setsid nohup ros2 run rover_multi link_probe --ros-args -r __ns:=/robot1 -r __node:=link_probe_peer \
  -p target:=$ROBOT2_HOST -p topic:=explore/link_probe_peer > $D/robot1_probe_peer.log 2>&1 < /dev/null &
echo $! > $D/robot1_probe_peer.pid
sleep 10
rsync -a ~/experiment_kit/robot2_run.sh ~/experiment_kit/robot2_stop.sh ~/experiment_kit/ros_cleanup.sh ~/experiment_kit/motors_off.py ${ROBOT2_USER}@${ROBOT2_HOST}:experiment_kit/ 2>/dev/null || \
  { $SSH 'mkdir -p ~/experiment_kit'; rsync -a ~/experiment_kit/robot2_run.sh ~/experiment_kit/robot2_stop.sh ~/experiment_kit/ros_cleanup.sh ~/experiment_kit/motors_off.py ${ROBOT2_USER}@${ROBOT2_HOST}:experiment_kit/; }
# robot1 record: everything robot1 has + what it RECEIVED from robot2 (team topics, odom, probes).
# robot2's map is recorded on robot2 only (sending it here would load the Wi-Fi being measured).
setsid nohup ros2 bag record -o $D/robot1_bag \
  -e '^/(team/.*|tf|tf_static|rosout|robot1/(explore/.*|map|odom|plan|cmd_vel|depth_scan|object_detector/detections|object_mapper/(detections_3d|objects))|robot2/(odom|explore/link_probe.*))$' \
  > $D/robot1_bag.log 2>&1 < /dev/null &
echo $! > $D/robot1_bag.pid
# robot2: the call returns once robot2_run.sh has started everything (all its jobs detach from ssh)
if ! timeout 60 $SSH "bash ~/experiment_kit/robot2_run.sh $RUN $VLM $R1_IP ${DRY:-0}" < /dev/null; then
  echo "!! robot2 did not start: stopping robot1 again"; bash ~/experiment_kit/stop_pair.sh $RUN; exit 1; fi
# robot1 takes its heading from the IMU only: a failed BNO085 makes the whole run useless
sleep 6
if grep -q "Failed to initialize BNO08X" $D/robot1_launch.log; then
  echo "!! robot1 IMU (BNO085) failed to initialize: reseat its cable. Stopping the run."
  bash ~/experiment_kit/stop_pair.sh $RUN; exit 1; fi
# Debug windows on robot1's screen. xterm talks to X directly; gnome-terminal started from a script
# reported success without opening anything on this Jetson (2026-10-02).
export DISPLAY=${DISPLAY:-:1}
(setsid xterm -T "run $RUN: coordinator log" -geometry 160x30+0+0 -fa Monospace -fs 10 -e bash -c \
  "tail -n 200 -f $D/robot1_launch.log | grep --line-buffered -E 'team_coordinator|team_agent|object_mapper.*New|died|ERROR|Failed'" \
  >/dev/null 2>&1 &) || true
if [ "$DEBUG_RVIZ" = 1 ]; then
  (setsid xterm -T "team rviz" -geometry 100x10+0+600 -e bash -c 'bash ~/run_scripts/rviz_team.sh' >/dev/null 2>&1 &) || true
fi
echo "started $RUN -> $D   (stop: bash ~/experiment_kit/stop_pair.sh $RUN)"
