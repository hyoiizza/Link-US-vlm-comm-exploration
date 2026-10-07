#!/usr/bin/env python3
"""Paper tables 1-2 and figure 3 from the runs made with ~/experiment_kit.

    python3 paper_outputs.py paper.yaml       -> <output>/table1.md, table2.md, table2.csv,
                                                  runs_detail.csv, fig3_trajectories.png/.pdf,
                                                  found/<run>_<robot>.jpg (frame counted as "found": check by eye)

paper.yaml:
    persons: {P1: [8.0, 3.5]}                  # tape-measured from robot1's start mark (x forward, y left)
    runs: [~/experiment_runs/b1_r1, ...]        # method / VLM / person from each run's manifest.yaml
    time_limit_s: 900
    output: ~/experiment_runs/_paper
    platform: "..."; environment: "..."         # free text for table 1 (optional)

Definitions (all times on robot1's clock; robot2 times shifted by the clock offset in the manifest):
    start          first goal award of the coordinator
    found          YOLO (rover_object): the time of the 3rd detection of the same person object (object_mapper id,
                   its min_hits) on either robot, from each robot's own bag. With the run's tape-measured target
                   (manifest target: {x, y}, start_pair.sh --front/--left/--right) only detections within
                   target_radius_m (default 1.5) of it count (other people are bystanders); without it, the
                   first person not near a start, target position = mean of those detections in world. Without detection topics: first rtabmap frame in which the tape-measured
                   person (persons:) is in view, 0.5-6 m ahead and not behind nearer depth. cut = found or limit
    distance       sum of both robots' odometry path length start..cut, each from its OWN bag
    success        found before the time limit
    R_out (%)      time share start..cut with measured RSSI < Q_min or not associated, mean of the two
                   robots, each from its OWN bag (lost Wi-Fi messages cannot hide an outage)
    outages (#)    heartbeat gaps > 4 s start..cut: robot2's at robot1 + the coordinator's at robot2
    figure 3       per method (Baseline, Proposed) the successful run whose discovery time is closest to the
                   method median; path coloured by measured RSSI (>= Q_min normal, < Q_min warning);
                   dashed circle: distance at which the fitted path-loss model predicts Q_min
"""
import csv, glob, json, math, os, sys, warnings

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import Bag, TfSeries, db_frames, person_visible, DEFAULTS

warnings.simplefilter('ignore', RuntimeWarning)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

LABEL = {'frontier': 'Baseline', 'comm_aware': 'Comm-aware', 'semantic_only': 'Semantic-only', 'proposed': 'Proposed'}
ORDER = ['frontier', 'comm_aware', 'semantic_only', 'proposed']
ROBOTS = ('robot1', 'robot2')
COL = {'robot1': '#1f77b4', 'robot2': '#9467bd'}


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def load_run(d):
    d = os.path.expanduser(d)
    man = yaml.safe_load(open(os.path.join(d, 'manifest.yaml')))
    off = float(man.get('clock_offset_robot2_minus_robot1_s') or 0.0)
    bags = {'robot1': Bag(os.path.join(d, 'robot1_bag'))}
    r2 = os.path.join(d, 'robot2', 'robot2_bag')
    if os.path.isdir(r2):
        bags['robot2'] = Bag(r2)
    dbs = {'robot1': os.path.join(d, 'robot1.db'), 'robot2': os.path.join(d, 'robot2', 'robot2.db')}
    params = {}
    for f in ('team.yaml', 'world.yaml', 'vlm.yaml'):
        p = os.path.join(d, 'params', f)
        if os.path.exists(p):
            params[f] = yaml.safe_load(open(p))
    return dict(dir=d, man=man, off={'robot1': 0.0, 'robot2': off}, bags=bags, dbs=dbs, params=params)


def find_key(tree, key):
    """First value of `key` anywhere in a nested params dict."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == key:
                return v
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def world_track(bag, robot, off):
    """[(t robot1 clock, x, y)] of the robot in world from its own odom + tf."""
    tf = TfSeries(bag)
    out = []
    for t, m in bag.rows(f'/{robot}/odom'):
        p = m.pose.pose.position
        T = tf.world_odom(robot, stamp(m)) @ np.array([p.x, p.y, 0.0, 1.0])
        out.append((t - off, T[0], T[1]))
    return np.array(out).reshape(-1, 3), tf


def path_len(P, t0, t1):
    Q = P[(P[:, 0] >= t0) & (P[:, 0] <= t1)]
    return float(np.hypot(*np.diff(Q[:, 1:3], axis=0).T).sum()) if len(Q) > 1 else 0.0


def rssi_series(bag, robot, off):
    """[(t, x, y, rssi or nan)] the robot measured itself (link_quality it published, from its own bag)."""
    out = []
    for t, m in bag.rows('/team/link_quality'):
        if m.robot == robot:
            out.append((t - off, m.position.x, m.position.y, m.rssi_dbm if m.connected else np.nan))
    return np.array(out).reshape(-1, 4)


def r_out(S, t0, t1, q_min):
    s = S[(S[:, 0] >= t0) & (S[:, 0] <= t1)]
    if len(s) == 0:
        return math.nan
    dt = np.minimum(np.diff(np.r_[s[:, 0], min(t1, s[-1, 0] + 0.5)]), 2.0)
    bad = np.isnan(s[:, 3]) | (s[:, 3] < q_min)
    return float((dt * bad).sum() / dt.sum())


def hb_gaps(bag, robot, off, t0, t1, gap=4.0):
    ts = sorted(t - off for t, m in bag.rows('/team/heartbeat') if m.robot == robot)
    ts = [t for t in ts if t0 <= t <= t1]
    if not ts:
        return 1 if t1 - t0 > gap else 0
    g = np.diff([t0] + ts + [t1])
    return int((g > gap).sum())


def find_person(run, px, py, cfg, t0, t1, tracks):
    best = None
    for robot in ROBOTS:
        db = run['dbs'][robot]
        if not os.path.exists(db) or robot not in run['bags']:
            continue
        c, frames = db_frames(db)
        tf = tracks[robot][1]
        off = run['off'][robot]
        for nid, st, Tob, cal in frames:
            t = st - off
            if t < t0 or t > t1 or (best and t >= best[0]):
                continue
            vis, dist, uv = person_visible(c, nid, tf.world_odom(robot, st) @ Tob @ cal['local'], cal, px, py, cfg)
            if vis:
                best = (t, robot, nid, dist, uv, c)
                break
    return best


IGNORE_NEAR_START_M = 1.5   # people this close to a robot's start are the experimenters, not the target


def find_person_yolo(run, t0, t1, hits=3, target=None, radius=1.5):
    """(t, robot, (x, y) world) of the first person confirmed by `hits` detections of one object id, or None.

    The second value is False when no robot recorded any detection topic (fall back to geometry)."""
    best, any_topic = None, False
    for robot, bag in run['bags'].items():
        rows = bag.rows(f'/{robot}/object_mapper/detections_3d')
        if rows or any(n == f'/{robot}/object_mapper/detections_3d' for c in bag.dbs
                       for (n,) in c.execute('select name from topics')):
            any_topic = True
        tf = TfSeries(bag)
        W = tf.world_map.get(robot, np.eye(4))
        seen = {}
        for t, m in rows:
            t -= run['off'][robot]
            if t < t0 or t > t1:
                continue
            for d in m.detections:
                if not d.results or d.results[0].hypothesis.class_id != 'person':
                    continue
                p = d.bbox.center.position
                w = W @ np.array([p.x, p.y, p.z, 1.0])
                starts = [(0.0, 0.0)] + [(M[0, 3], M[1, 3]) for M in tf.world_map.values()]
                if min(math.hypot(w[0] - sx, w[1] - sy) for sx, sy in starts) < IGNORE_NEAR_START_M:
                    continue
                if target is not None and math.hypot(w[0] - target[0], w[1] - target[1]) > radius:
                    continue
                seen.setdefault(d.id, []).append((t, w[0], w[1]))
                if len(seen[d.id]) == hits:
                    pts = np.array(seen[d.id])
                    if best is None or t < best[0]:
                        best = (t, robot, (float(pts[:, 1].mean()), float(pts[:, 2].mean())))
                    break
    return best, any_topic


def fit_path_loss(series, ap=(0.0, 0.0)):
    """P0, n of RSSI = P0 - 10 n log10(d) over every connected sample of every run (AP = world origin)."""
    pts = np.concatenate([s for s in series if len(s)]) if series else np.zeros((0, 4))
    pts = pts[np.isfinite(pts[:, 3])]
    if len(pts) < 20:
        return math.nan, math.nan, math.nan, 0
    d = np.maximum(np.hypot(pts[:, 1] - ap[0], pts[:, 2] - ap[1]), 1.0)
    A = np.c_[np.ones(len(d)), -10 * np.log10(d)]
    (p0, n), *_ = np.linalg.lstsq(A, pts[:, 3], rcond=None)
    return float(p0), float(n), float(np.std(pts[:, 3] - A @ [p0, n])), len(pts)


def tkey(r):
    """The target a run searched for: its measured position, else the person id of paper.yaml."""
    return (r['target_x'], r['target_y']) if r.get('target_source') == 'measured' else r['person']


def med_iqr(v, fmt='{:.1f}'):
    v = np.array([x for x in v if x == x], dtype=float)
    if len(v) == 0:
        return '–'
    q1, m, q3 = np.percentile(v, [25, 50, 75])
    return f'{fmt.format(m)} [{fmt.format(q1)}–{fmt.format(q3)}]'


def main():
    spec = yaml.safe_load(open(sys.argv[1]))
    out = os.path.expanduser(spec.get('output', '~/experiment_runs/_paper'))
    os.makedirs(os.path.join(out, 'found'), exist_ok=True)
    cfg = dict(DEFAULTS)
    limit = float(spec.get('time_limit_s', 900))
    persons = {k: tuple(v) for k, v in (spec.get('persons') or {}).items()}
    rows, runs = [], []
    for d in spec['runs']:
        run = load_run(d); runs.append(run)
        man = run['man']; name = str(man['run'])
        team = run['params'].get('team.yaml', {})
        q_min = float(find_key(team, 'q_min') or -70.0)
        print(f'== {name} ({man["method"]})', flush=True)
        tracks = {r: world_track(run['bags'][r], r, run['off'][r]) for r in ROBOTS if r in run['bags']}
        awards = [t for t, m in run['bags']['robot1'].rows('/team/auction/award') if any(a.task.id for a in m.assignments)]
        t0 = awards[0] if awards else tracks['robot1'][0][0, 0]
        t_end = max(P[-1, 0] for P, _ in tracks.values() if len(P))
        t1 = min(t_end, t0 + limit)
        person = man.get('person')
        px, py = persons.get(str(person), (math.nan, math.nan))
        tgt = man.get('target') or {}
        target = (float(tgt['x']), float(tgt['y'])) if tgt.get('x') is not None else None
        if target:
            px, py = target
        yolo, has_det = find_person_yolo(run, t0, t1, target=target, radius=float(spec.get('target_radius_m', 1.5)))
        fx = fy = math.nan
        if has_det:
            found = (yolo[0], yolo[1], None, None, None, None) if yolo else None
            if yolo:
                fx, fy = yolo[2]
                if not target:
                    px, py = fx, fy
        else:
            found = find_person(run, px, py, cfg, t0, t1, tracks) if person in persons else None
        cut = found[0] if found else t1
        if found and found[2] is not None:
            t, robot, nid, dist, (u, v), c = found
            img = cv2.imdecode(np.frombuffer(c.execute('select image from Data where id=?', (nid,)).fetchone()[0], np.uint8), 1)
            cv2.circle(img, (u, v), 18, (0, 0, 255), 3)
            cv2.putText(img, f'{name} {robot} node {nid} t={t - t0:.1f}s', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out, 'found', f'{name}_{robot}.jpg'), img)
        rssi = {r: rssi_series(run['bags'][r], r, run['off'][r]) for r in tracks}
        ro = [r_out(rssi[r], t0, cut, q_min) for r in rssi]
        outages = (hb_gaps(run['bags']['robot1'], 'robot2', run['off']['robot2'], t0, cut) if 'robot2' in run['bags'] else 0) + \
                  (hb_gaps(run['bags']['robot2'], 'robot1', run['off']['robot1'], t0, cut) if 'robot2' in run['bags'] else 0)
        row = dict(run=name, method=man['method'], use_vlm=man['use_vlm'], person=person,
                   success=int(found is not None), T_found_s=round(cut - t0, 1) if found else math.nan,
                   found_by=found[1] if found else '',
                   dist_found_m=round(sum(path_len(P, t0, cut) for P, _ in tracks.values()), 1),
                   R_out_pct=round(100 * float(np.nanmean(ro)), 2) if ro else math.nan,
                   outages=outages, duration_s=round(t1 - t0, 1), q_min=q_min)
        row['found_method'] = 'yolo' if has_det else 'geometry'
        row['target_x'], row['target_y'] = (round(px, 2), round(py, 2)) if px == px else ('', '')
        row['target_source'] = 'measured' if target else ('yolo' if px == px else '')
        row['found_x'], row['found_y'] = (round(fx, 2), round(fy, 2)) if fx == fx else ('', '')
        rows.append(row); run.update(row=row, tracks=tracks, rssi=rssi, t0=t0, cut=cut, q_min=q_min, person_xy=(px, py))
        print('   ' + ', '.join(f'{k} {v}' for k, v in row.items() if k not in ('run', 'method')))
    # table 2
    with open(os.path.join(out, 'runs_detail.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    t2 = ['| 방법 | 발견 시간 (s) ↓ | 발견까지 거리 (m) ↓ | 성공률 (%) ↑ | R_out (%) ↓ | 단절 (회) ↓ |',
          '| --- | --- | --- | --- | --- | --- |']
    csvrows = []
    for meth in ORDER:
        g = [r for r in rows if r['method'] == meth]
        if not g:
            t2.append(f'| {LABEL[meth]} | – | – | – | – | – |'); continue
        succ = 100 * np.mean([r['success'] for r in g])
        cells = [med_iqr([r['T_found_s'] for r in g]), med_iqr([r['dist_found_m'] for r in g]),
                 f'{succ:.0f} ({sum(r["success"] for r in g)}/{len(g)})',
                 med_iqr([r['R_out_pct'] for r in g]), med_iqr([r['outages'] for r in g], '{:.0f}')]
        name = f'**{LABEL[meth]}**' if meth == 'proposed' else LABEL[meth]
        t2.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        csvrows.append([LABEL[meth], len(g)] + cells)
    ns = sorted({sum(1 for r in rows if r['method'] == m and tkey(r) == p) for m in ORDER for p in {tkey(r) for r in rows}} - {0})
    t2.insert(0, f'**표 2. 방법별 실험 결과** (중앙값 [IQR], 조건당 {"/".join(map(str, ns)) or "–"}회)\n')
    open(os.path.join(out, 'table2.md'), 'w').write('\n'.join(t2) + '\n')
    with open(os.path.join(out, 'table2.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'n', 'T_found_s', 'dist_found_m', 'success', 'R_out_pct', 'outages']); w.writerows(csvrows)
    # table 1
    team = runs[0]['params'].get('team.yaml', {}); vlm = runs[0]['params'].get('vlm.yaml', {})
    ap = (float(find_key(team, 'ap_x') or 0.0), float(find_key(team, 'ap_y') or 0.0))
    p0, n, sd, npts = fit_path_loss([s for run in runs for s in run['rssi'].values()], ap)
    q_min = runs[0]['q_min']
    d_qmin = 10 ** ((p0 - q_min) / (10 * n)) if n == n else math.nan
    cover = []
    for run in runs:
        b = run['bags']['robot1'].rows('/robot1/map')
        if b:
            m = b[-1][1]; cover.append(int((np.asarray(m.data) >= 0).sum()) * m.info.resolution ** 2)
    reps = sorted({sum(1 for r in rows if r['method'] == m and tkey(r) == p) for m in ORDER for p in {tkey(r) for r in rows}} - {0})
    t1md = [
        '**표 1. 실험 설정**\n', '| 범주 | 항목 | 값 |', '| --- | --- | --- |',
        f'| 플랫폼 | 로버, 센서 | {spec.get("platform", "[기재 필요]")} |',
        f'| 환경 | 공간 규모, 목표 배치 | {spec.get("environment", "")} 탐사 면적 중앙값 {np.median(cover):.0f} m² (robot1 지도); 목표 '
        + (', '.join(f'{k} ({v[0]:.1f}, {v[1]:.1f}) m' for k, v in persons.items()) if persons else
           ', '.join(sorted({f'({r["target_x"]}, {r["target_y"]}) m' for r in rows if r["target_x"] != ""}))
           + (' (실측, robot1 출발점 기준)' if all(r['target_source'] == 'measured' for r in rows) else ' (YOLO 검출 위치)')) + ' |',
        f'| 통신 | AP 위치 / 경로 손실 모델 | ({ap[0]:.2f}, {ap[1]:.2f}) m (robot1 출발점 기준) / P0 = {p0:.1f} dBm, n = {n:.2f} (σ = {sd:.1f} dB, 표본 {npts}) |',
        f'|  | Q_min | {q_min:.0f} dBm (모델상 반경 {d_qmin:.1f} m) |',
        f'| VLM | 모델 | SigLIP2 base-patch16-224 |',
        f'| 투영 | [Z_min, Z_max] / N_min | [{find_key(vlm, "min_range_m")}, {find_key(vlm, "max_range_m")}] m / {find_key(vlm, "min_reference_crops")} |',
        f'| 효용 | α / λ | {find_key(team, "alpha")} / {find_key(team, "lambda")} |',
        f'| 프로토콜 | 반복 횟수, 제한 시간 | 조건당 {"/".join(map(str, reps))}회 (방법 {len({r["method"] for r in rows})} × 목표 {len({tkey(r) for r in rows})}), {limit:.0f} s |']
    open(os.path.join(out, 'table1.md'), 'w').write('\n'.join(t1md) + '\n')
    # figure 3
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, meth, tag in zip(axs, ('frontier', 'proposed'), ('(a)', '(b)')):
        g = [r for r in runs if r['row']['method'] == meth]
        ok = [r for r in g if r['row']['success']] or g
        if not ok:
            ax.set_title(f'{tag} {LABEL[meth]}: no run'); ax.axis('off'); continue
        med = np.nanmedian([r['row']['T_found_s'] for r in ok]) if any(r['row']['success'] for r in ok) else None
        run = min(ok, key=lambda r: abs(r['row']['T_found_s'] - med)) if med == med and med is not None else ok[0]
        b = run['bags']['robot1'].rows('/robot1/map')
        if b:
            m = b[-1][1]; gm = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
            img = np.full(gm.shape, np.nan); img[(gm >= 0) & (gm < 50)] = 0.9; img[gm >= 50] = 0.0
            ox, oy, r = m.info.origin.position.x, m.info.origin.position.y, m.info.resolution
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.5, interpolation='nearest',
                      extent=(ox, ox + gm.shape[1] * r, oy, oy + gm.shape[0] * r))
        xs, ys = [0.0], [0.0]
        for robot in ROBOTS:
            if robot not in run['tracks']:
                continue
            P = run['tracks'][robot][0]; P = P[(P[:, 0] >= run['t0']) & (P[:, 0] <= run['cut'])]
            S = run['rssi'][robot]; S = S[(S[:, 0] >= run['t0']) & (S[:, 0] <= run['cut'])]
            if len(P):
                ax.plot(P[:, 1], P[:, 2], '-', color=COL[robot], lw=1.0, alpha=0.6, label=robot)
                xs += [P[:, 1].min(), P[:, 1].max()]; ys += [P[:, 2].min(), P[:, 2].max()]
            if len(S):
                good = S[:, 3] >= run['q_min']
                ax.scatter(S[good, 1], S[good, 2], s=9, c='#2ca02c', zorder=3)
                ax.scatter(S[~good, 1], S[~good, 2], s=12, c='#d62728', zorder=4)
        ax.scatter([], [], s=9, c='#2ca02c', label=f'RSSI ≥ {run["q_min"]:.0f} dBm')
        ax.scatter([], [], s=12, c='#d62728', label=f'RSSI < {run["q_min"]:.0f} dBm / lost')
        ax.plot(ap[0], ap[1], '^', color='k', ms=10, label='AP')
        ax.plot(0, 0, 'P', color='0.3', ms=8, label='robot1 start')
        px, py = run['person_xy']
        if px == px:
            ax.plot(px, py, '*', color='orange', mec='k', ms=16, label='target'); xs.append(px); ys.append(py)
        if d_qmin == d_qmin:
            ax.add_patch(plt.Circle(ap, d_qmin, fill=False, ls='--', color='k', lw=1.0))
        if d_qmin == d_qmin and d_qmin < 30:          # show the whole Q_min circle when it is near the paths
            xs += [ap[0] - d_qmin, ap[0] + d_qmin]; ys += [ap[1] - d_qmin, ap[1] + d_qmin]
        ax.set_xlim(min(xs) - 2, max(xs) + 2); ax.set_ylim(min(ys) - 2, max(ys) + 2)
        rw = run['row']
        ax.set_title(f'{tag} {LABEL[meth]}  ({rw["run"]}: {rw["T_found_s"]} s, {rw["dist_found_m"]} m)', fontsize=10)
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig3_trajectories.png'), dpi=200); fig.savefig(os.path.join(out, 'fig3_trajectories.pdf'))
    print(open(os.path.join(out, 'table1.md')).read()); print(open(os.path.join(out, 'table2.md')).read())
    print(f'-> {out}')


if __name__ == '__main__':
    main()
