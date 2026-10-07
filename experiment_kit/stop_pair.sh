#!/usr/bin/env bash
# Stop a run on both robots (recorders first) and collect robot2's files into robot1's run folder.
source ~/experiment_kit/config.sh
RUN=$1; D=$RUNS_DIR/$RUN; [ -d $D ] || { echo "no run $RUN"; exit 1; }
stop() { [ -f $D/$1.pid ] && kill -INT -$(ps -o pgid= $(cat $D/$1.pid) | tr -d ' ') 2>/dev/null; }
stop robot1_bag
$SSH "bash ~/experiment_kit/robot2_stop.sh $RUN" || echo "!! robot2 not reachable: stop it there with bash ~/experiment_kit/robot2_stop.sh $RUN"
sleep 3
stop robot1_probe_ap; stop robot1_probe_peer; stop robot1_launch; sleep 8
bash ~/experiment_kit/ros_cleanup.sh            # orphaned nodes of the launch
ls $D/robot1_bag | grep -q metadata.yaml && echo "robot1 bag closed" || echo "robot1 bag NOT closed"
echo "end: $(date -Iseconds)" >> $D/manifest.yaml
mkdir -p $D/robot2 && rsync -a ${ROBOT2_USER}@${ROBOT2_HOST}:experiment_runs/$RUN/ $D/robot2/ && echo "robot2 files -> $D/robot2"
du -sh $D
