#!/usr/bin/env python3
"""One-line health check per run: did both robots map, plan, drive and talk?
    python3 run_health.py ~/experiment_runs/<run> [...]"""
import collections, glob, math, os, re, sqlite3, sys
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import Bag, TfSeries
from paper_outputs import world_track, rssi_series

def topic_counts(bagdir):
    n = collections.Counter()
    for f in glob.glob(os.path.join(bagdir, '*.db3')):
        c = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
        for name, k in c.execute('select t.name,count(*) from topics t join messages m on m.topic_id=t.id group by t.id'):
            n[name] += k
    return n

def log_counts(path):
    out = collections.Counter()
    if not os.path.exists(path):
        return out
    txt = open(path, errors='ignore').read()
    shut = txt.find('signal_handler(SIGINT')
    pre = txt if shut < 0 else txt[:shut]
    out['died_before_stop'] = len(re.findall(r'process has died', pre))
    out['rtab_nodata'] = txt.count('Did not receive data since')
    out['no_map'] = txt.count("no map received")
    out['no_path'] = txt.count('no valid path found')
    out['no_progress'] = txt.count('Failed to make progress')
    out['imu_fail'] = txt.count('Failed to get product IDs')
    return out

for d in sys.argv[1:]:
    d = os.path.expanduser(d).rstrip('/')
    man = yaml.safe_load(open(os.path.join(d, 'manifest.yaml')))
    tgt = man.get('target') or {}
    bags = {'robot1': os.path.join(d, 'robot1_bag'), 'robot2': os.path.join(d, 'robot2', 'robot2_bag')}
    print(f"== {man['run']} ({man['method']}, vlm {man['use_vlm']}) target {tgt.get('x')},{tgt.get('y')}")
    log1 = open(os.path.join(d, 'robot1_launch.log'), errors='ignore').read() if os.path.exists(os.path.join(d, 'robot1_launch.log')) else ''
    tasks = collections.Counter((m.group(1), 'ok' if 'succeeded' in m.group(2) else 'fail')
                                for m in re.finditer(r'(robot\d): task \S+ (succeeded|failed[^\n]*)', log1))
    sel = re.findall(r'Selection \d+: (robot\d) ->', log1)
    for robot, bd in bags.items():
        if not glob.glob(os.path.join(bd, '*.db3')):
            print(f'   {robot}: NO BAG'); continue
        n = topic_counts(bd)
        b = Bag(bd)
        try:
            P, _ = world_track(b, robot, 0.0)
            dur = P[-1, 0] - P[0, 0]; L = float(np.hypot(*np.diff(P[:, 1:3], axis=0).T).sum())
            ext = f'x {P[:,1].min():.1f}..{P[:,1].max():.1f} y {P[:,2].min():.1f}..{P[:,2].max():.1f}'
        except Exception as e:
            dur, L, ext = math.nan, math.nan, str(e)[:40]
        S = rssi_series(b, robot, 0.0); q = S[:, 3] if len(S) else np.array([np.nan])
        lc = log_counts(os.path.join(d, 'robot1_launch.log' if robot == 'robot1' else 'robot2/robot2_launch.log'))
        W = TfSeries(b).world_map.get(robot, np.eye(4))
        near = 0; ppl = 0
        for t, m in b.rows(f'/{robot}/object_mapper/detections_3d'):
            for det in m.detections:
                if det.results and det.results[0].hypothesis.class_id == 'person':
                    p = det.bbox.center.position; w = W @ np.array([p.x, p.y, p.z, 1]); ppl += 1
                    if tgt.get('x') is not None and math.hypot(w[0] - tgt['x'], w[1] - tgt['y']) <= 1.5:
                        near += 1
        print(f"   {robot}: {dur:5.0f} s, {L:5.1f} m ({ext}); map msgs {n.get(f'/{robot}/map', 0)}; "
              f"goals sel {sel.count(robot)} ok {tasks[(robot,'ok')]} fail {tasks[(robot,'fail')]}; "
              f"RSSI {np.nanmin(q):.0f}..{np.nanmax(q):.0f}; person dets {ppl} (near target {near})")
        print(f"      log: " + ', '.join(f'{k} {v}' for k, v in lc.items() if v))
    if not glob.glob(os.path.join(bags['robot1'], '*.db3')):
        continue
    b1 = Bag(bags['robot1'])
    T0 = b1.rows('/robot1/odom')[0][0] if b1.rows('/robot1/odom') else 0
    hb = [t - T0 for t, m in b1.rows('/team/heartbeat') if m.robot == 'robot2']
    gaps = [round(c - a, 1) for a, c in zip(hb, hb[1:]) if c - a > 4]
    ann = {m.auction_id: t for t, m in b1.rows('/team/auction/announce')}
    bd = collections.defaultdict(list); pl = collections.defaultdict(list)
    for t, m in b1.rows('/team/auction/bid'):
        if m.auction_id in ann: bd[m.robot].append(t - ann[m.auction_id])
        pl[m.robot] += [i.path_planned for i in m.items]
    print(f"   robot2 heartbeat at robot1: first {hb[0] if hb else float('nan'):.0f} s, gaps>4s {len(gaps)}; "
          + '; '.join(f"{r} bid delay med {np.median(v):.1f} s, planned {100*np.mean(pl[r]):.0f}%" for r, v in sorted(bd.items())))
