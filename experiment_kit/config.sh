# Experiment settings shared by the kit scripts (edit here, not in the scripts).
ROBOT2_USER=gitae
ROBOT2_HOST=${ROBOT2_HOST:-192.168.0.202}   # robot2 fixed IP on linkus (nmcli manual, 2026-10-02)
EXPECT_SSID=${EXPECT_SSID:-linkus}
RUNS_DIR=$HOME/experiment_runs
DEBUG_RVIZ=${DEBUG_RVIZ:-0}                  # 1: RViz on robot1's screen (adds Wi-Fi load: robot2 map)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5 ${ROBOT2_USER}@${ROBOT2_HOST}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
