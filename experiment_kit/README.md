# Two-robot experiment kit (run everything from robot1)

## Before a session
1. Both robots on the experiment Wi-Fi (`linkus`), power save off, robot2 IP in `config.sh` (`ROBOT2_HOST`).
2. `bash ~/experiment_kit/sync_robot2.sh`  - push code changes to robot2 and rebuild there.
3. Place robot1 on the start mark; robot2 at its fixed offset (`rover_multi/config/world.yaml`, now 0.6 m left).
   The person (or mannequin) stays at its place for the whole run.

## One run
```
bash ~/experiment_kit/start_pair.sh <run_name> <method> <use_vlm 0|1> [person_id] ["note"]
#   method: frontier | comm_aware | semantic_only | proposed
bash ~/experiment_kit/stop_pair.sh <run_name>        # recorders first, then robots; collects robot2's files
```
`DEBUG_RVIZ=1 bash start_pair.sh ...` also opens RViz on robot1 (adds Wi-Fi load: avoid in measured runs).

Everything of a run lands in `~/experiment_runs/<run_name>/`:
`manifest.yaml` (method, VLM, person, SSIDs, power save, clock offset robot2-robot1), `params/` (config snapshot),
`robot1_bag/` (robot1 + what it received), `robot1.db`, `robot2/` (robot2's own bag, db, logs), launch/probe logs.

## Analysis
```
python3 ~/experiment_eval/compare_runs.py ~/experiment_runs/<run> [...]   # trajectories + communication
python3 ~/experiment_eval/metrics.py runs.yaml                            # T_rel, D_total, R_out (person positions)
```
For metrics.py use `bag: ~/experiment_runs/<run>/robot1_bag` and
`dbs: {robot1: .../robot1.db, robot2: .../robot2/robot2.db}`.
