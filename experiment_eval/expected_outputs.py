#!/usr/bin/env python3
"""EXPECTED (hypothetical) tables 1-2 and figure 3, to get a feel for the paper before the runs.
Nothing here is measured: ranges from the single-robot run (0.10 m/s), the offline replay (S changes
~25 % of choices) and the link measurement (P0 -24 dBm, n 4.04, loss onset ~-73 dBm)."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/experiment_eval'))
from metrics import Bag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.expanduser('~/experiment_runs/_expected_20261002')
P0, N, QMIN = -24.0, 4.04, -70.0
AP = np.array([-0.25, 0.30])
R_Q = 10 ** ((P0 - QMIN) / (10 * N))
TARGET = np.array([10.5, 2.4])

t1 = f"""**표 1. 실험 설정** (예정 값; [ ]는 실험 후 확정)

| 범주 | 항목 | 값 |
| --- | --- | --- |
| 플랫폼 | 로버, 센서 | 6륜 로커-보기 로버 2대 (Jetson Orin Nano 8GB, RPLIDAR C1, Orbbec Gemini 335; robot1만 BNO085 IMU) |
| 환경 | 공간 규모, 목표 배치 | 실내 복도 약 35 × 15 m [실측 지도로 확정]; 목표 1명, 출발점에서 약 11 m 떨어진 방 안 [위치 확정] |
| 통신 | AP 위치 / 경로 손실 모델 | robot1 출발점 옆 ({AP[0]:.2f}, {AP[1]:.2f}) m / P0 = {P0:.0f} dBm, n = {N:.2f} (10/1 실측) |
|  | Q_min | {QMIN:.0f} dBm (모델상 반경 {R_Q:.1f} m) |
| VLM | 모델 | SigLIP2 base-patch16-224 |
| 투영 | [Z_min, Z_max] / N_min | [0.5, 6] m / 30 |
| 효용 | α / λ | 0.5 / 1.0 |
| 프로토콜 | 반복 횟수, 제한 시간 | 방법 × 목표당 [3~5]회, 900 s |
"""
rows = [('Baseline', '450 (300–600)', '75 (50–100)', '60–90', '18 (10–25)', '2 (1–3)'),
        ('Comm-aware', '520 (350–700)', '80 (50–110)', '50–80', '2 (0–5)', '0 (0–1)'),
        ('Semantic-only', '400 (250–550)', '70 (45–95)', '60–90', '18 (10–25)', '2 (1–3)'),
        ('**Proposed**', '430 (280–600)', '72 (50–100)', '60–90', '2 (0–5)', '0 (0–1)')]
t2 = ['**표 2. 방법별 실험 결과 — 예상치 (가설, 측정값 아님)** (중앙값 (예상 범위))\n',
      '| 방법 | 발견 시간 (s) ↓ | 발견까지 거리 (m) ↓ | 성공률 (%) ↑ | R_out (%) ↓ | 단절 (회) ↓ |',
      '| --- | --- | --- | --- | --- | --- |'] + ['| ' + ' | '.join(r) + ' |' for r in rows]
t2 += ['', '근거: 평균 탐사 속도 0.10 m/s (10/1 단독 주행), S가 골 선택을 바꾸는 비율 약 25% (오프라인 재생),',
       f'RSSI < {QMIN:.0f} dBm 반경 약 {R_Q:.0f} m·−90 dBm 이하 단절 (10/1 통신 측정). 실험 후 실측값으로 교체.']
open(f'{OUT}/table1_expected.md', 'w').write(t1)
open(f'{OUT}/table2_expected.md', 'w').write('\n'.join(t2) + '\n')

def rssi(p):
    return P0 - 10 * N * np.log10(np.maximum(np.hypot(p[:, 0] - AP[0], p[:, 1] - AP[1]), 1.0))

def path(*pts, step=0.25):
    out = []
    for a, b in zip(pts, pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        n = max(int(np.hypot(*(b - a)) / step), 1)
        out += [a + (b - a) * k / n for k in range(n)]
    return np.array(out + [np.array(pts[-1], float)])

# illustrative routes in the corridor of the 10/1 map (y ~ 0 corridor, branch at x ~ 10-13)
routes = {
    'Baseline': {'robot1': path((0, 0), (6, 0.2), (10.5, 0.3), (10.5, 1.6), (10.5, 2.4)),
                 'robot2': path((0, -1.2), (8, -0.6), (16, 0.2), (24, 0.4), (30, 0.6), (34, 0.5))},
    'Proposed': {'robot1': path((0, 0), (5, 0.2), (9.5, 0.3), (10.5, 1.5), (10.5, 2.4)),
                 'robot2': path((0, -1.2), (6, -0.8), (10, -2.0), (12.5, -2.3))}}
b = Bag(os.path.expanduser('~/bags_recovered/0151/rec_0.db3'))
m = b.rows('/robot1/map')[-1][1]
g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
img = np.full(g.shape, np.nan); img[(g >= 0) & (g < 50)] = 0.9; img[g >= 50] = 0.0
ox, oy, r = m.info.origin.position.x, m.info.origin.position.y, m.info.resolution
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, (tag, meth) in zip(axs, (('(a)', 'Baseline'), ('(b)', 'Proposed'))):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.5, interpolation='nearest',
              extent=(ox, ox + g.shape[1] * r, oy, oy + g.shape[0] * r))
    for robot, P in routes[meth].items():
        q = rssi(P); good = q >= QMIN
        ax.plot(P[:, 0], P[:, 1], '-', color='#1f77b4' if robot == 'robot1' else '#9467bd', lw=1, alpha=0.6, label=robot)
        ax.scatter(P[good, 0], P[good, 1], s=8, c='#2ca02c', zorder=3)
        ax.scatter(P[~good, 0], P[~good, 1], s=10, c='#d62728', zorder=4)
    ax.scatter([], [], s=8, c='#2ca02c', label=f'RSSI ≥ {QMIN:.0f} dBm (model)')
    ax.scatter([], [], s=10, c='#d62728', label=f'RSSI < {QMIN:.0f} dBm (model)')
    ax.plot(*AP, '^', color='k', ms=10, label='AP')
    ax.plot(*TARGET, '*', color='orange', mec='k', ms=16, label='target')
    ax.add_patch(plt.Circle(AP, R_Q, fill=False, ls='--', color='k', lw=1))
    ax.set_xlim(-4, 36); ax.set_ylim(-14, 14); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(f'{tag} {meth} — EXPECTED (illustrative, not measured)', fontsize=10)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.legend(fontsize=7, loc='lower right')
fig.text(0.5, 0.01, 'Paths are hand-drawn illustrations on the real 10/1 map; RSSI colours from the fitted path-loss model, not measurements.',
         ha='center', fontsize=8, color='0.3')
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{OUT}/fig3_expected.png', dpi=180); fig.savefig(f'{OUT}/fig3_expected.pdf')
print(open(f'{OUT}/table1_expected.md').read()); print('\n'.join(t2)); print('->', OUT)
