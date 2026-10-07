#!/usr/bin/env bash
# Emergency / "I forgot the run name" stop of BOTH robots, from robot1 (both on the same Wi-Fi).
# Recorders first (so the bags close), then every ROS process. Needs nothing installed on robot2.
source ~/experiment_kit/config.sh
STOP='pkill -INT -f "[r]os2 bag record"; sleep 4;
      pkill -INT -f "[r]os2 (launch|run)"; sleep 6;
      pkill -TERM -f "[/]ros2_ws/install/|[/]opt/ros/humble/lib/"; sleep 2;
      pkill -KILL -f "[/]ros2_ws/install/|[/]opt/ros/humble/lib/";
      [ -e /dev/lx16a ] && python3 ~/experiment_kit/motors_off.py;
      echo "$(hostname): stopped, ROS processes left: $(ps -eo args | grep -cE "[/]ros2_ws/install/|[/]opt/ros/humble/lib/")"'
echo "== robot2 ($ROBOT2_HOST)"
rsync -a --timeout=10 ~/experiment_kit/motors_off.py ~/experiment_kit/ros_cleanup.sh ${ROBOT2_USER}@${ROBOT2_HOST}:experiment_kit/ 2>/dev/null
if ssh -o BatchMode=yes -o ConnectTimeout=5 ${ROBOT2_USER}@${ROBOT2_HOST} "$STOP"; then :; else
  echo "!! robot2 not reachable from here. Is robot1 on the same Wi-Fi (linkus)?"
  echo "   On robot2 itself run:  bash ~/experiment_kit/robot2_stop.sh <run>   (or cut the motor power)"; fi
echo "== robot1"
bash -c "$STOP"
# collect robot2's files of every run that robot1 knows but has not collected yet
for d in ~/experiment_runs/*/; do
  run=$(basename $d)
  [ -f $d/manifest.yaml ] || continue
  ls $d/robot2/robot2_bag/metadata.yaml >/dev/null 2>&1 && continue
  mkdir -p $d/robot2
  rsync -a --timeout=20 ${ROBOT2_USER}@${ROBOT2_HOST}:experiment_runs/$run/ $d/robot2/ 2>/dev/null && echo "collected robot2 files of $run"
done
