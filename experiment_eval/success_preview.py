#!/usr/bin/env python3
"""What a successful result looks like: (a) the REAL robot1 trajectory of the 10/1 01:51 run in the
figure-3 format, (b) an illustrative successful Proposed run with the criteria marked, (c) where each
method should land in (R_out, discovery time) if the hypothesis holds (expected values, not measured).
The 01:51 measured RSSI is unusable (hotspot moved with the robot, ~-23 dBm all run), so colours use
the fitted path-loss model around the configured AP."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/experiment_eval'))
from metrics import Bag
from paper_outputs import world_track
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from expected_outputs import P0, N, QMIN, AP, R_Q, TARGET, rssi, path, OUT

b = Bag('~/bags_recovered/0151/rec_0.db3')
P, _ = world_track(b, 'robot1', 0.0)
P[:, 0] -= P[0, 0]
m = b.rows('/robot1/map')[-1][1]
g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
img = np.full(g.shape, np.nan); img[(g >= 0) & (g < 50)] = 0.9; img[g >= 50] = 0.0
ox, oy, r = m.info.origin.position.x, m.info.origin.position.y, m.info.resolution
ext = (ox, ox + g.shape[1] * r, oy, oy + g.shape[0] * r)
length = float(np.hypot(*np.diff(P[:, 1:3], axis=0).T).sum())

def base(ax, title):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.5, interpolation='nearest', extent=ext)
    ax.plot(*AP, '^', color='k', ms=10, label='AP')
    ax.add_patch(plt.Circle(AP, R_Q, fill=False, ls='--', color='k', lw=1))
    ax.set_xlim(-4, 36); ax.set_ylim(-14, 16); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

def colored(ax, X, Y, label, size=6):
    q = rssi(np.c_[X, Y]); good = q >= QMIN
    ax.scatter(X[good], Y[good], s=size, c='#2ca02c', zorder=3)
    ax.scatter(X[~good], Y[~good], s=size, c='#d62728', zorder=4)
    return (~good).mean()

fig = plt.figure(figsize=(17, 5.6))
ax = fig.add_subplot(1, 3, 1)
base(ax, f'(a) REAL: robot1 alone, 10/1 01:51 (proposed)\n{P[-1,0]:.0f} s, {length:.0f} m, no target placed')
ro = colored(ax, P[::10, 1], P[::10, 2], 'robot1')
ax.plot(P[:, 1], P[:, 2], '-', color='#1f77b4', lw=0.8, alpha=0.6, label='robot1 (measured)')
ax.plot(P[0, 1], P[0, 2], 'o', color='w', mec='k', ms=7, label='start')
ax.scatter([], [], s=8, c='#2ca02c', label=f'≥ {QMIN:.0f} dBm (model)'); ax.scatter([], [], s=8, c='#d62728', label=f'< {QMIN:.0f} dBm (model)')
ax.text(15, 12, f'R_out (model) = {ro*100:.0f}%\n→ left the Q_min zone:\nthis is the FAILURE pattern\nthe constraint should prevent', fontsize=8,
        bbox=dict(fc='w', ec='#d62728'))
ax.legend(fontsize=7, loc='lower right')

ax = fig.add_subplot(1, 3, 2)
base(ax, '(b) ILLUSTRATIVE: what a successful Proposed run looks like')
r1 = path((0, 0), (5, 0.2), (9.5, 0.3), (10.5, 1.5), (10.5, 2.4))
r2 = path((0, -1.2), (6, -0.8), (10, -2.0), (12.5, -2.3))
for R, c, lab in ((r1, '#1f77b4', 'robot1'), (r2, '#9467bd', 'robot2')):
    ax.plot(R[:, 0], R[:, 1], '-', color=c, lw=1.2, label=lab); colored(ax, R[:, 0], R[:, 1], lab, 10)
ax.plot(*TARGET, '*', color='orange', mec='k', ms=18, label='target (YOLO hit)')
ax.legend(fontsize=7, loc='lower right')
kw = dict(fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8), bbox=dict(fc='w', ec='0.5'))
ax.annotate('① target found (3rd YOLO hit)\n   → success, T and distance\n      measured up to here', TARGET, (14, 8), **kw)
ax.annotate('② all points green:\n   R_out ≈ 0, no outages', (12.5, -2.3), (16, -8), **kw)
ax.annotate('③ robots take different\n   frontiers (no duplication)', (6, -0.8), (-3, -11), **kw)
ax.annotate('④ goes straight to the room\n   with high S (person-like view)', (9.5, 0.3), (-3, 9), **kw)

ax = fig.add_subplot(1, 3, 3)
E = {'Baseline': (18, 450, '#7f7f7f'), 'Comm-aware': (2, 520, '#2ca02c'),
     'Semantic-only': (18, 400, '#ff7f0e'), 'Proposed': (2, 430, '#d62728')}
ax.axvspan(-2, 5, color='#2ca02c', alpha=0.08); ax.axhspan(380, 450, color='#1f77b4', alpha=0.06)
for k, (x, y, c) in E.items():
    ax.errorbar(x, y, yerr=[[60], [80]], fmt='o', color=c, ms=9, capsize=3)
    ax.text(x + 1, y + 8, k, fontsize=9, color=c, weight='bold' if k == 'Proposed' else None)
ax.annotate('', (3, 425), (17, 445), arrowprops=dict(arrowstyle='->', color='0.4'))
ax.text(7, 470, 'Q constraint:\nR_out ↓', fontsize=8, color='0.3')
ax.annotate('', (3, 435), (3, 510), arrowprops=dict(arrowstyle='->', color='0.4'))
ax.text(-1.5, 560, 'S: T ↓', fontsize=8, color='0.3')
ax.set_xlim(-2, 25); ax.set_ylim(300, 640)
ax.set_xlabel('R_out [%]  (time below Q_min)  ← better'); ax.set_ylabel('discovery time [s]  ← better')
ax.set_title('(c) EXPECTED: success = Proposed in the lower-left\n(R_out like Comm-aware, T close to Semantic-only)', fontsize=9)
ax.grid(alpha=0.3)
fig.text(0.5, 0.01, '(a) measured trajectory, RSSI colours from the path-loss model (P0 -24 dBm, n 4.04, AP at the configured position); '
         '(b),(c) are illustrations / hypotheses, not measurements.', ha='center', fontsize=8, color='0.3')
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{OUT}/success_preview.png', dpi=160); fig.savefig(f'{OUT}/success_preview.pdf')
print('R_out model', ro, 'length', length, '->', OUT)
