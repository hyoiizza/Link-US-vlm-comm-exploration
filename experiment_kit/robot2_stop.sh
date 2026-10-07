#!/usr/bin/env bash
# Runs ON robot2. Arg: run_name. Recorder first (closes the bag), then probes and launch.
D=~/experiment_runs/$1
stop() { [ -f $D/$1.pid ] && kill -INT -$(ps -o pgid= $(cat $D/$1.pid) | tr -d ' ') 2>/dev/null; }
stop robot2_bag; sleep 4
stop robot2_probe_ap; stop robot2_probe_peer; stop robot2_launch; sleep 8
bash ~/experiment_kit/ros_cleanup.sh
ls $D/robot2_bag >/dev/null 2>&1 && ls $D/robot2_bag | grep -q metadata.yaml && echo "robot2 bag closed" || echo "robot2 bag NOT closed"
