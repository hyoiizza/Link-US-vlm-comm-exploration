#!/usr/bin/env python3
"""Per-run experiment metrics from the robot1 bag and each robot's rtabmap DB.

    python3 metrics.py runs.yaml            -> results/runs.csv, results/summary.txt,
                                               results/<run>_found.jpg (frame to check by eye)

Coordinates: world = robot1's start pose (x forward, y left), which is the same physical
frame in every run as long as robot1 starts on the marked spot with the same heading. Person
positions are measured once with a tape from that mark.

Metrics (README of this folder has the definitions for the paper):
  T_rel     time from the first goal award to the first camera frame (either robot) in which
            the person is in view, in range and not hidden behind nearer depth
  D_total   sum of both robots' odometry path length (until found / until the end)
  R_out_rssi  fraction of time a robot's measured RSSI < q_min or not associated (mean of robots)
  R_out_hb    fraction of time robot2's heartbeat (as received on robot1) had gaps > outage_gap_s
  plus: success, explored area, goal switches, Nav2 failures, mean S / G / D / Q of chosen goals
"""
import csv, glob, math, os, re, sqlite3, struct, sys, warnings
from collections import defaultdict

import numpy as np
import cv2
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rvl import decode_rvl

warnings.simplefilter("ignore", RuntimeWarning)   # nan means of single runs

DEFAULTS = dict(q_min=-70.0, outage_gap_s=4.0, time_limit_s=900.0, min_range_m=0.5, max_range_m=6.0,
                occlusion_margin_m=0.5, person_heights_m=[0.5, 0.9, 1.3], start='first_award')


# ---- bag (sqlite, also works on recovered files) ---------------------------------------
class Bag:
    def __init__(self, path):
        path = os.path.expanduser(path)
        files = sorted(glob.glob(os.path.join(path, '*.db3'))) if os.path.isdir(path) else [path]
        if not files:
            raise SystemExit(f'no .db3 in {path}')
        self.dbs = [sqlite3.connect(f'file:{f}?mode=ro', uri=True) for f in files]

    def rows(self, name):
        out = []
        for c in self.dbs:
            tp = {i: t for i, n, t in c.execute('select id,name,type from topics') if n == name}
            for tid, typ in tp.items():
                T = get_message(typ)
                out += [(t * 1e-9, deserialize_message(d, T)) for t, d in
                        c.execute('select timestamp,data from messages where topic_id=? order by timestamp', (tid,))]
        return sorted(out, key=lambda x: x[0])


def quat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_matrix(tr):
    T = np.eye(4); T[:3, :3] = quat(tr.transform.rotation)
    T[:3, 3] = [tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z]
    return T


class TfSeries:
    """map->odom per robot over time, world->map (static) per robot."""
    def __init__(self, bag):
        self.map_odom = defaultdict(list)
        for _, m in bag.rows('/tf'):
            for tr in m.transforms:
                if tr.header.frame_id.endswith('/map') and tr.child_frame_id.endswith('/odom'):
                    s = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    self.map_odom[tr.header.frame_id.split('/')[0]].append((s, tf_matrix(tr)))
        self.world_map = {}
        for _, m in bag.rows('/tf_static'):
            for tr in m.transforms:
                if tr.header.frame_id == 'world' and tr.child_frame_id.endswith('/map'):
                    self.world_map[tr.child_frame_id.split('/')[0]] = tf_matrix(tr)
        for v in self.map_odom.values():
            v.sort(key=lambda x: x[0])

    def world_odom(self, robot, t):
        seq = self.map_odom.get(robot, [])
        mo = np.eye(4)
        if seq:
            i = max(np.searchsorted([s for s, _ in seq], t, side='right') - 1, 0)
            mo = seq[i][1]
        return self.world_map.get(robot, np.eye(4)) @ mo


# ---- rtabmap DB -------------------------------------------------------------------------
def db_frames(path):
    """[(node id, stamp, T_odom_base 4x4, calib dict)] in time order."""
    c = sqlite3.connect(f'file:{os.path.expanduser(path)}?mode=ro', uri=True)
    out = []
    for nid, stamp, pose, cal in c.execute(
            'select n.id, n.stamp, n.pose, d.calibration from Node n join Data d on n.id=d.id '
            'where d.calibration is not null order by n.stamp'):
        h = struct.unpack_from('<11i', cal, 0); o = 44
        K = np.frombuffer(cal, '<f8', h[6], o); o += 8 * (h[6] + h[7] + h[8] + h[9])
        local = np.eye(4); local[:3, :] = np.frombuffer(cal, '<f4', 12, o).reshape(3, 4)
        Tob = np.eye(4); Tob[:3, :] = np.frombuffer(pose, '<f4').reshape(3, 4)
        out.append((nid, stamp, Tob, dict(fx=K[0], fy=K[4], cx=K[2], cy=K[5], w=h[4], h=h[5], local=local)))
    return c, out


def person_visible(c, nid, T_world_cam, cal, px, py, cfg):
    """(visible, distance, (u, v)) of the person (vertical segment at px, py) in this frame."""
    Tcw = np.linalg.inv(T_world_cam)
    best = None
    for z in cfg['person_heights_m']:
        p = Tcw @ np.array([px, py, z, 1.0])
        X, Y, Z = p[:3]
        if not cfg['min_range_m'] <= Z <= cfg['max_range_m']:
            continue
        u, v = cal['cx'] + cal['fx'] * X / Z, cal['cy'] + cal['fy'] * Y / Z
        if not (0 <= u < cal['w'] and 0 <= v < cal['h']):
            continue
        best = best or []
        best.append((Z, int(u), int(v)))
    if not best:
        return False, None, None
    blob = c.execute('select depth from Data where id=?', (nid,)).fetchone()[0]
    depth = decode_rvl(blob)
    for Z, u, v in best:
        if depth is None:
            return True, Z, (u, v)                      # no depth: no occlusion test
        win = depth[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].astype(np.float32) * 0.001
        win = win[win > 0]
        if len(win) == 0 or np.median(win) >= Z - cfg['occlusion_margin_m']:
            return True, Z, (u, v)
    return False, None, None


# ---- metrics ------------------------------------------------------------------------------
def path_length(odom, t0, t1):
    pts = np.array([(m.pose.pose.position.x, m.pose.pose.position.y) for t, m in odom if t0 <= t <= t1])
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def outage_fraction(samples, t0, t1, bad):
    """Time-weighted fraction of [t0, t1] where bad(sample) (each sample holds until the next, max 2 s)."""
    s = [(t, m) for t, m in samples if t0 <= t <= t1]
    if not s:
        return math.nan
    tot = badt = 0.0
    for (t, m), (tn, _) in zip(s, s[1:] + [(min(t1, s[-1][0] + 1.0), None)]):
        dt = min(tn - t, 2.0); tot += dt; badt += dt * bad(m)
    return badt / tot if tot else math.nan


def heartbeat_outage(hb, robot, t0, t1, gap):
    ts = [t for t, m in hb if m.robot == robot and t0 <= t <= t1]
    if len(ts) < 2:
        return math.nan
    g = np.diff([t0] + ts + [t1])
    return float(g[g > gap].sum() / (t1 - t0))


SEL = re.compile(r'Selection \d+: (robot\d) -> \S+ \(([-\d.]+), ([-\d.]+)\) U=([-\d.]+) S=([-\d.]+) G=([-\d.]+) D=([-\d.]+) Q=(\S+)')


def run_metrics(run, cfg, persons, outdir):
    bag = Bag(run['bag'])
    tf = TfSeries(bag)
    awards = [t for t, m in bag.rows('/team/auction/award') if any(a.task.id for a in m.assignments)]
    t_start = awards[0] if awards else bag.rows('/robot1/odom')[0][0]
    t_bag_end = max(t for t, _ in bag.rows('/team/heartbeat')) if bag.rows('/team/heartbeat') else t_start
    t_end = min(t_bag_end, t_start + cfg['time_limit_s'])
    px, py = persons[run['person']]
    # T_rel: first frame of either robot that sees the person
    found = None
    for robot, db in run['dbs'].items():
        c, frames = db_frames(db)
        for nid, stamp, Tob, cal in frames:
            if stamp < t_start or stamp > t_end:
                continue
            Twc = tf.world_odom(robot, stamp) @ Tob @ cal['local']
            vis, dist, uv = person_visible(c, nid, Twc, cal, px, py, cfg)
            if vis:
                if found is None or stamp < found[0]:
                    found = (stamp, robot, nid, dist, uv, c)
                break
    t_found = found[0] if found else None
    t_rel = (t_found - t_start) if found else math.nan
    if found:
        _, robot, nid, dist, (u, v), c = found
        img = cv2.imdecode(np.frombuffer(c.execute('select image from Data where id=?', (nid,)).fetchone()[0], np.uint8), 1)
        cv2.circle(img, (u, v), 18, (0, 0, 255), 3)
        cv2.putText(img, f"{run['name']} {robot} node {nid} T_rel {t_rel:.1f}s dist {dist:.1f}m", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(outdir, f"{run['name']}_found.jpg"), img)
    t_cut = t_found or t_end
    robots = list(run['dbs'])
    odom = {r: bag.rows(f'/{r}/odom') for r in robots}
    link = bag.rows('/team/link_quality'); hb = bag.rows('/team/heartbeat')
    bad = lambda m: (not m.connected) or m.rssi_dbm < cfg['q_min']
    r_rssi = [outage_fraction([(t, m) for t, m in link if m.robot == r], t_start, t_end, bad) for r in robots]
    others = [r for r in robots if r != 'robot1']
    r_hb = [heartbeat_outage(hb, r, t_start, t_end, cfg['outage_gap_s']) for r in others]
    # explored area (sum of each robot's known cells) at the cut and at the end
    def area(t_at):
        tot = 0.0
        for r in robots:
            maps = [(t, m) for t, m in bag.rows(f'/{r}/map') if t <= t_at]
            if maps:
                m = maps[-1][1]; tot += int((np.asarray(m.data) >= 0).sum()) * m.info.resolution ** 2
        return tot
    status = [m for t, m in bag.rows('/team/task_status') if t_start <= t <= t_end]
    sels = [SEL.search(m.msg) for t, m in bag.rows('/rosout') if m.name == 'team_coordinator' and t_start <= t <= t_cut]
    sels = [s for s in sels if s]
    mean = lambda k: float(np.mean([float(s.group(k)) for s in sels])) if sels else math.nan
    qv = [float(s.group(8)) for s in sels if s.group(8) not in ('nan',)]
    return dict(
        run=run['name'], method=run['method'], person=run['person'], success=int(found is not None),
        T_rel_s=round(t_rel, 1), found_by=found[1] if found else '', found_dist_m=round(found[3], 2) if found else '',
        D_total_found_m=round(sum(path_length(odom[r], t_start, t_cut) for r in robots), 1),
        D_total_end_m=round(sum(path_length(odom[r], t_start, t_end) for r in robots), 1),
        R_out_rssi=round(float(np.nanmean(r_rssi)), 4) if r_rssi else math.nan,
        R_out_hb=round(float(np.nanmean(r_hb)), 4) if r_hb and not all(math.isnan(x) for x in r_hb) else math.nan,
        area_found_m2=round(area(t_cut), 1), area_end_m2=round(area(t_end), 1),
        goal_switches=sum(1 for m in status if m.reason.startswith('replaced')),
        nav_failures=sum(1 for m in status if m.status == 3),
        selections=len(sels), mean_S=round(mean(5), 3), mean_G=round(mean(6), 3), mean_D=round(mean(7), 3),
        mean_Q=round(float(np.mean(qv)), 1) if qv else math.nan,
        duration_s=round(t_end - t_start, 1))


def main():
    spec = yaml.safe_load(open(sys.argv[1]))
    cfg = {**DEFAULTS, **(spec.get('defaults') or {})}
    persons = {k: tuple(v) for k, v in spec['persons'].items()}
    outdir = os.path.expanduser(spec.get('output', 'results')); os.makedirs(outdir, exist_ok=True)
    rows = []
    for run in spec['runs']:
        print(f"== {run['name']} ({run['method']})", flush=True)
        r = run_metrics(run, cfg, persons, outdir); rows.append(r)
        print('   ' + ', '.join(f'{k} {v}' for k, v in r.items() if k not in ('run', 'method')))
    with open(os.path.join(outdir, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [f'{"method":14s} n  success  T_rel[s]      D_total(found)[m]  R_out_rssi    R_out_hb']
    for meth in dict.fromkeys(r['method'] for r in rows):
        g = [r for r in rows if r['method'] == meth]
        ms = lambda k: (np.nanmean([float(x[k]) for x in g]), np.nanstd([float(x[k]) for x in g]))
        t, d, ro, rh = ms('T_rel_s'), ms('D_total_found_m'), ms('R_out_rssi'), ms('R_out_hb')
        lines.append(f'{meth:14s} {len(g):<2d} {np.mean([x["success"] for x in g]):6.0%}   {t[0]:6.1f}±{t[1]:<5.1f}  '
                     f'{d[0]:6.1f}±{d[1]:<5.1f}      {ro[0]:.3f}±{ro[1]:.3f}  {rh[0]:.3f}±{rh[1]:.3f}')
    open(os.path.join(outdir, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'-> {outdir}/runs.csv, summary.txt, *_found.jpg')


if __name__ == '__main__':
    main()
