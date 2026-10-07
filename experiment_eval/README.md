# Experiment metrics

`python3 metrics.py runs.yaml` -> `results/runs.csv` (one row per run), `results/summary.txt`
(mean ± std per method), `results/<run>_found.jpg` (the frame that counted as "found": check by eye).

## Recording (robot1, every run)
One bag on robot1 holds both robots (/tf and /team/* are global, robot2 odom/map come over Wi-Fi):
```
ros2 bag record -o ~/bags/$(date +%Y%m%d_%H%M%S)_<method>_<person> -e '^/(team/.*|tf|tf_static|rosout|robot[12]/(explore/.*|map|odom|plan|cmd_vel))$'
```
After the run copy robot2's rtabmap DB (`~/maps/..._robot2_....db`) to robot1.
Stop the recorder first (Ctrl+C) so the bag is closed properly.

## Setup rules
- robot1 starts on the marked spot with the same heading every run (world frame = that pose);
  robot2 at its fixed offset (`config/world.yaml` robot2.initial_pose).
- The person stays in place for the whole run (the check assumes a static person).
- Person position measured once with a tape from robot1's mark: x forward, y left.

## Definitions
- **T_rel**: time from the first goal award to the first rtabmap frame of either robot in which the
  person (points at 0.5 / 0.9 / 1.3 m height) projects into the image, lies 0.5-6 m in front of the
  camera and is not hidden (depth at that pixel not nearer than distance - 0.5 m).
  Not found within `time_limit_s` -> failure (success = 0, T_rel = nan).
- **D_total**: sum of both robots' /odom path lengths, until found and until the end.
- **R_out_rssi**: time-weighted fraction with measured RSSI < q_min or not associated (mean of robots).
- **R_out_hb**: fraction of time with robot2 heartbeat gaps > outage_gap_s as received on robot1.
- Also: explored area (sum of known cells of both maps), goal switches, Nav2 failures,
  mean S / G / D / Q of the coordinator's chosen goals (from its log).

Camera poses come from each DB node's odometry pose x map->odom (bag /tf at the node stamp)
x world->map (bag /tf_static) x the node's camera calibration. Depth is rtabmap RVL (`rvl.py`).
