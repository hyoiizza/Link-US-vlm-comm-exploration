#!/usr/bin/env python3
"""Trajectory and communication comparison of two-robot runs made with ~/experiment_kit.

    python3 compare_runs.py ~/experiment_runs/<run> [<run> ...]   -> ~/experiment_runs/_compare/
        runs_traj_comm.csv          one row per run (trajectory + communication metrics)
        summary_by_method.txt       mean ± std per method
        <run>_traj.png              both robots' world trajectories on both maps
        all_runs_traj.png           every run side by side (grouped by method)

Run folder layout (start_pair.sh / stop_pair.sh):
    manifest.yaml, robot1_bag/ (robot1 + what it received from robot2), robot2/robot2_bag/ (robot2 local)

Trajectory: world pose = world->map (tf_static) x map->odom (tf) x odom (odom topic), per robot.
Communication, per direction:
    robot2 -> robot1   heartbeat, candidates, bid, task_status, link_quality of robot2:
                       sent = robot2's own bag, received = robot1 bag (matched by header stamp)
    robot1 -> robot2   coordinator announce / award: sent = robot1 bag, received = robot2 bag
    delivery = received / sent, latency = receive time - header stamp (corrected by the clock
    offset measured at the start, manifest clock_offset_robot2_minus_robot1_s)
    heartbeat outage = time in gaps > 4 s; link probes: RTT / loss to the AP and to the other robot
    bandwidth = bytes/s of robot2-originated messages received on robot1
"""
import csv, math, os, sys, warnings
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import Bag, TfSeries

warnings.simplefilter('ignore', RuntimeWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAP = 4.0
R2_TOPICS = ['/team/heartbeat', '/team/candidates', '/team/auction/bid', '/team/task_status', '/team/link_quality']
R1_TOPICS = ['/team/auction/announce', '/team/auction/award']
COLORS = {'robot1': '#1f77b4', 'robot2': '#d62728'}


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def world_traj(bag, robot):
    tf = TfSeries(bag)
    pts = []
    for t, m in bag.rows(f'/{robot}/odom'):
        p = m.pose.pose.position
        T = tf.world_odom(robot, stamp(m)) @ np.array([p.x, p.y, 0.0, 1.0])
        pts.append((t, T[0], T[1]))
    return np.array(pts), tf


def last_map(bag, robot, tf):
    maps = bag.rows(f'/{robot}/map')
    if not maps:
        return None
    m = maps[-1][1]
    g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    W = tf.world_map.get(robot, np.eye(4))
    ox = W[0, 3] + m.info.origin.position.x
    oy = W[1, 3] + m.info.origin.position.y
    return g, m.info.resolution, ox, oy


def path_len(P):
    return float(np.hypot(*np.diff(P[:, 1:3], axis=0).T).sum()) if len(P) > 1 else 0.0


def overlap(P, Q, radius=1.0):
    """Fraction of Q's path (sampled every 0.25 m) within radius of P's path: redundant coverage."""
    if len(P) < 2 or len(Q) < 2:
        return math.nan
    q = Q[::max(1, len(Q) // 400), 1:3]
    p = P[::max(1, len(P) // 2000), 1:3]
    d = np.sqrt(((q[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
    return float((d <= radius).mean())


def coverage_m2(maps, res=0.1):
    """Known area of the union of both maps in world [m^2] (0.1 m grid)."""
    cells = set()
    for mp in maps:
        if mp is None:
            continue
        g, r, ox, oy = mp
        ii, jj = np.nonzero(g >= 0)
        xs, ys = ox + (jj + 0.5) * r, oy + (ii + 0.5) * r
        cells.update(zip(np.floor(xs / res).astype(int).tolist(), np.floor(ys / res).astype(int).tolist()))
    return len(cells) * res * res


def link_stats(bag, topic):
    rows = [m.data for _, m in bag.rows(topic)]
    if not rows:
        return {}
    a = np.array(rows, dtype=float)                  # sent, received, loss, rtt_mean, rtt_max, rssi
    sent, recv = np.nansum(a[:, 0]), np.nansum(a[:, 1])
    rtt = a[:, 3][np.isfinite(a[:, 3])]
    rssi = a[:, 5][np.isfinite(a[:, 5])]
    return dict(loss=1 - recv / sent if sent else math.nan,
                rtt_med=float(np.median(rtt)) if len(rtt) else math.nan,
                rtt_p95=float(np.percentile(rtt, 95)) if len(rtt) else math.nan,
                rssi_med=float(np.median(rssi)) if len(rssi) else math.nan,
                rssi_min=float(rssi.min()) if len(rssi) else math.nan,
                not_assoc_s=float(np.isnan(a[:, 5]).sum()))


def delivery(sent_bag, recv_bag, topic, key, offset, who=None):
    """(sent, received, delivery, latency median, p95) for one topic and direction."""
    def items(bag):
        out = {}
        for t, m in bag.rows(topic):
            if who and getattr(m, 'robot', who) != who:
                continue
            out.setdefault(key(m), (t, m))
        return out
    s, r = items(sent_bag), items(recv_bag)
    if not s:
        return 0, 0, math.nan, math.nan, math.nan
    got = [k for k in s if k in r]
    lat = np.array([r[k][0] - (stamp(s[k][1]) - offset) for k in got]) if got else np.array([])
    return (len(s), len(got), len(got) / len(s),
            float(np.median(lat)) * 1000 if len(lat) else math.nan,
            float(np.percentile(lat, 95)) * 1000 if len(lat) else math.nan)


def hb_outage(bag, robot, t0, t1):
    ts = sorted(t for t, m in bag.rows('/team/heartbeat') if m.robot == robot and t0 <= t <= t1)
    if len(ts) < 2:
        return math.nan, 0, math.nan
    g = np.diff([t0] + ts + [t1])
    return float(g[g > GAP].sum() / (t1 - t0)), int((g > GAP).sum()), float(g.max())


def bandwidth(bag, topics, robot, t0, t1):
    tot = 0
    for c in bag.dbs:
        tp = {i: n for i, n, _ in c.execute('select id,name,type from topics')}
        for tid, name in tp.items():
            if name in topics:
                tot += c.execute('select sum(length(data)) from messages where topic_id=?', (tid,)).fetchone()[0] or 0
            elif name.startswith(f'/{robot}/'):
                tot += c.execute('select sum(length(data)) from messages where topic_id=?', (tid,)).fetchone()[0] or 0
    return tot / max(t1 - t0, 1e-3) / 1024.0


def draw(ax, maps, trajs, lw=1.6, legend=True):
    """Maps (free light grey, occupied black) and trajectories; axes limited to explored area + 1.5 m."""
    xs, ys = [], []
    for mp in maps:
        if mp is None:
            continue
        g, r, ox, oy = mp
        img = np.full(g.shape, np.nan); img[(g >= 0) & (g < 50)] = 0.88; img[g >= 50] = 0.0
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.6,
                  extent=(ox, ox + g.shape[1] * r, oy, oy + g.shape[0] * r), interpolation='nearest')
        ii, jj = np.nonzero(g >= 0)
        if len(ii):
            xs += [ox + jj.min() * r, ox + (jj.max() + 1) * r]; ys += [oy + ii.min() * r, oy + (ii.max() + 1) * r]
    for robot, P in trajs:
        if len(P):
            ax.plot(P[:, 1], P[:, 2], '-', color=COLORS[robot], lw=lw, label=f'{robot} ({path_len(P):.1f} m)')
            ax.plot(P[0, 1], P[0, 2], 'o', color=COLORS[robot], ms=7)
            ax.plot(P[-1, 1], P[-1, 2], 's', color=COLORS[robot], ms=7)
            xs += [P[:, 1].min(), P[:, 1].max()]; ys += [P[:, 2].min(), P[:, 2].max()]
    if xs:
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5); ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    if legend:
        ax.legend(loc='best', fontsize=8)


def analyse(run_dir, outdir):
    man = yaml.safe_load(open(os.path.join(run_dir, 'manifest.yaml')))
    name = str(man['run'])
    b1 = Bag(os.path.join(run_dir, 'robot1_bag'))
    r2dir = os.path.join(run_dir, 'robot2', 'robot2_bag')
    b2 = Bag(r2dir) if os.path.isdir(r2dir) else None
    off = float(man.get('clock_offset_robot2_minus_robot1_s') or 0.0)
    P1, tf1 = world_traj(b1, 'robot1')
    P2, tf2 = world_traj(b2, 'robot2') if b2 else (np.zeros((0, 3)), None)
    t0 = min(x for x in [P1[0, 0] if len(P1) else None, P2[0, 0] - off if len(P2) else None] if x is not None)
    t1 = max(x for x in [P1[-1, 0] if len(P1) else None, P2[-1, 0] - off if len(P2) else None] if x is not None)
    maps = [last_map(b1, 'robot1', tf1), last_map(b2, 'robot2', tf2) if b2 else None]
    row = dict(run=name, method=man['method'], use_vlm=man['use_vlm'], person=man.get('person'),
               duration_s=round(t1 - t0, 1), clock_offset_ms=round(off * 1000, 1),
               path_robot1_m=round(path_len(P1), 1), path_robot2_m=round(path_len(P2), 1),
               D_total_m=round(path_len(P1) + path_len(P2), 1),
               overlap_r2_on_r1=round(overlap(P1, P2), 3), coverage_m2=round(coverage_m2(maps), 1))
    # communication
    if b2:
        for topic in R2_TOPICS:
            s, r, dlv, lm, lp = delivery(b2, b1, topic, lambda m: (stamp(m), getattr(m, 'robot', '')), off, 'robot2')
            k = topic.split('/')[-1]
            row.update({f'r2to1_{k}_sent': s, f'r2to1_{k}_deliv': round(dlv, 4) if dlv == dlv else dlv,
                        f'r2to1_{k}_lat_med_ms': round(lm, 1), f'r2to1_{k}_lat_p95_ms': round(lp, 1)})
        for topic in R1_TOPICS:
            s, r, dlv, lm, lp = delivery(b1, b2, topic, lambda m: m.auction_id, -off)
            k = topic.split('/')[-1]
            row.update({f'r1to2_{k}_sent': s, f'r1to2_{k}_deliv': round(dlv, 4) if dlv == dlv else dlv,
                        f'r1to2_{k}_lat_med_ms': round(lm, 1), f'r1to2_{k}_lat_p95_ms': round(lp, 1)})
        frac, n, mx = hb_outage(b1, 'robot2', t0, t1)
        row.update(hb_outage_frac_r2_at_r1=round(frac, 4), hb_gaps_r2_at_r1=n, hb_max_gap_s=round(mx, 1))
    for bag, robot in ((b1, 'robot1'), (b2, 'robot2')):
        if bag is None:
            continue
        for suffix, topic in (('ap', f'/{robot}/explore/link_probe'), ('peer', f'/{robot}/explore/link_probe_peer')):
            st = link_stats(bag, topic)
            for k, v in st.items():
                row[f'{robot}_{suffix}_{k}'] = round(v, 3) if isinstance(v, float) else v
    row['bw_from_r2_KBps'] = round(bandwidth(b1, R2_TOPICS, 'robot2', t0, t1), 2)
    # figure
    fig, ax = plt.subplots(figsize=(7, 7))
    draw(ax, maps, (('robot1', P1), ('robot2', P2)))
    ax.set_title(f"{name}: {man['method']}, VLM {'on' if str(man['use_vlm']) in ('1', 'True', 'true') else 'off'}")
    ax.set_xlabel('world x [m]'); ax.set_ylabel('world y [m]')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_traj.png'), dpi=130); plt.close(fig)
    return row, (name, man['method'], P1, P2, maps)


def main():
    runs = [os.path.expanduser(r) for r in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    outdir = os.path.join(os.path.dirname(runs[0].rstrip('/')), '_compare')
    os.makedirs(outdir, exist_ok=True)
    rows, figs = [], []
    for r in runs:
        print(f'== {r}', flush=True)
        row, fig = analyse(r, outdir)
        rows.append(row); figs.append(fig)
        for k in ('duration_s', 'D_total_m', 'overlap_r2_on_r1', 'coverage_m2', 'hb_outage_frac_r2_at_r1',
                  'r2to1_heartbeat_deliv', 'r2to1_heartbeat_lat_med_ms', 'robot1_peer_rtt_med', 'robot1_peer_loss',
                  'bw_from_r2_KBps'):
            print(f'   {k}: {row.get(k)}')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(outdir, 'runs_traj_comm.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    lines = []
    num = [k for k in keys if all(isinstance(r.get(k), (int, float)) for r in rows if k in r) and k not in ('use_vlm',)]
    for meth in dict.fromkeys(r['method'] for r in rows):
        g = [r for r in rows if r['method'] == meth]
        lines.append(f'## {meth} (n={len(g)})')
        for k in num:
            v = np.array([float(r[k]) for r in g if k in r], dtype=float)
            if len(v):
                lines.append(f'  {k:40s} {np.nanmean(v):10.3f} ± {np.nanstd(v):.3f}')
    open(os.path.join(outdir, 'summary_by_method.txt'), 'w').write('\n'.join(lines) + '\n')
    # all runs, grouped by method
    figs.sort(key=lambda x: (x[1], x[0]))
    n = len(figs); cols = min(n, 3); rws = math.ceil(n / cols)
    fig, axs = plt.subplots(rws, cols, figsize=(5 * cols, 5 * rws), squeeze=False)
    for ax, (name, meth, P1, P2, maps) in zip(axs.ravel(), figs):
        draw(ax, maps, (('robot1', P1), ('robot2', P2)), lw=1.2, legend=False)
        ax.set_title(f'{name} ({meth})', fontsize=9)
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'all_runs_traj.png'), dpi=110); plt.close(fig)
    print(f'-> {outdir}')


if __name__ == '__main__':
    main()
