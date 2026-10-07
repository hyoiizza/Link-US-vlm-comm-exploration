#!/usr/bin/env bash
# Stop every ROS node of this machine's workspace / ROS install that is still running (orphans of a
# killed launch). Matches by executable path, so it never kills the calling shell.
sel() { ps -eo pid=,args= | awk -v me=$$ '$1 != me && ($2 ~ /^\/opt\/ros\/humble\/lib\/|\/ros2_ws\/install\// || ($2 ~ /python3$/ && ($3 ~ /\/ros2_ws\/install\// || $3 ~ /^\/opt\/ros\/humble\/(bin|lib)\//)))' | awk '{print $1}'; }
n=$(sel | wc -l); [ "$n" = 0 ] && { echo "no ROS processes running"; exit 0; }
echo "stopping $n ROS processes"
sel | xargs -r kill -INT 2>/dev/null; sleep 6
sel | xargs -r kill -TERM 2>/dev/null; sleep 3
sel | xargs -r kill -KILL 2>/dev/null; sleep 1
echo "remaining: $(sel | wc -l)"
# Drive motors keep the last speed if the motor node died without sending 0: zero them directly.
[ -e /dev/lx16a ] && python3 ~/experiment_kit/motors_off.py 2>&1 | tail -1
