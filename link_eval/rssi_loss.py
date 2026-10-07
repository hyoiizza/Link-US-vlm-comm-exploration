"""RSSI vs packet loss / RTT from a bag with <robot>/explore/link_probe (+ /team/link_quality, /team/heartbeat).

    python3 rssi_loss.py <bag dir> [robot1]

Prints per 2 dB RSSI bin: windows, mean loss, windows with loss > 10 %, RTT p50/p95, distance
from the AP (= world origin, robot start); and heartbeat outages (gap > 4 s) for R_out.
"""
import math, sys, collections, warnings
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag = sys.argv[1]; robot = sys.argv[2] if len(sys.argv) > 2 else 'robot1'
r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
probe, pos, hb = [], [], collections.defaultdict(list)
tfpos, mo, ob = [], None, None   # robot position from /tf (map->odom x odom->base) when there is no link_quality
def yaw_xy(tr):
    q = tr.transform.rotation; th = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return th, tr.transform.translation.x, tr.transform.translation.y
while r.has_next():
    tp, data, t = r.read_next(); s = t * 1e-9
    if tp == '/tf':
        for tr in deserialize_message(data, get_message(types[tp])).transforms:
            if tr.header.frame_id == f'{robot}/map' and tr.child_frame_id == f'{robot}/odom': mo = yaw_xy(tr)
            elif tr.header.frame_id == f'{robot}/odom' and tr.child_frame_id == f'{robot}/base_link':
                ob = yaw_xy(tr)
                th, x0, y0 = mo or (0.0, 0.0, 0.0)
                tfpos.append((s, x0 + math.cos(th) * ob[1] - math.sin(th) * ob[2], y0 + math.sin(th) * ob[1] + math.cos(th) * ob[2]))
        continue
    if tp == f'/{robot}/explore/link_probe':
        probe.append((s, *deserialize_message(data, get_message(types[tp])).data))
    elif tp == '/team/link_quality':
        m = deserialize_message(data, get_message(types[tp]))
        if m.robot == robot: pos.append((s, m.position.x, m.position.y))
    elif tp == '/team/heartbeat':
        hb[deserialize_message(data, get_message(types[tp])).robot].append(s)
if not probe: sys.exit('no link_probe messages')
P = np.array(probe)          # t, sent, received, loss, rtt_mean, rtt_max, rssi
Q = np.array(pos) if pos else (np.array(tfpos) if tfpos else None)   # AP = world origin = robot1 start
dist = np.full(len(P), np.nan)
if Q is not None:
    i = np.clip(np.searchsorted(Q[:, 0], P[:, 0]), 0, len(Q) - 1); dist = np.hypot(Q[i, 1], Q[i, 2])
ok = P[:, 1] > 0
disc = np.isnan(P[:, 6])
print(f'{len(P)} windows ({P[-1,0]-P[0,0]:.0f} s), not associated {int(disc.sum())}, overall loss {1-P[ok,2].sum()/P[ok,1].sum():.3f}')
print(' RSSI bin   windows  mean loss  win loss>10%  RTT p50/p95 [ms]  dist to AP [m]')
for lo in range(int(np.nanmin(P[:, 6]) // 2 * 2), int(np.nanmax(P[:, 6])) + 2, 2):
    m = ok & (P[:, 6] >= lo) & (P[:, 6] < lo + 2)
    if not m.any(): continue
    loss = 1 - P[m, 2].sum() / P[m, 1].sum(); rtt = P[m, 4][~np.isnan(P[m, 4])]
    print(f' [{lo:4d},{lo+2:4d})  {m.sum():6d}   {loss:7.3f}    {np.mean(P[m,3] > 0.1):8.2f}     '
          f'{(np.percentile(rtt,50) if len(rtt) else math.nan):6.1f}/{(np.percentile(rtt,95) if len(rtt) else math.nan):6.1f}'
          f'      {np.nanmedian(dist[m]):5.1f}')
if disc.any(): print(f' not associated: {disc.sum()} windows, dist median {np.nanmedian(dist[disc]):.1f} m')
for rb, ts in hb.items():
    g = np.diff(np.array(ts)); out = g[g > 4.0]
    print(f'heartbeat {rb}: {len(ts)} msgs, outages >4 s: {len(out)} totalling {out.sum():.1f} s')
