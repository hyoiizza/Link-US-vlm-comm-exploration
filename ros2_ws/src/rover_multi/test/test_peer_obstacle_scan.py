import math

import numpy as np

from rover_multi.peer_obstacle_scan import disc_ranges


def beam(n, deg):
    return int(round((math.radians(deg) + math.pi) / (2 * math.pi / n))) % n


def test_peer_straight_ahead_and_left():
    r = disc_ranges(360, [(2.0, 0.0)], 0.42, 8.0)
    assert abs(r[beam(360, 0)] - 1.58) < 1e-6               # 2.0 - 0.42 on the centre beam
    hit = np.where(~np.isnan(r))[0]
    width = len(hit)                                        # 2*asin(0.42/2) = 24.2 deg
    assert 23 <= width <= 26
    assert np.isnan(r[beam(360, 90)]) and np.isnan(r[beam(360, 180)])
    r2 = disc_ranges(360, [(0.0, 1.2)], 0.42, 8.0)          # robot2 1.2 m to the left
    assert abs(r2[beam(360, 90)] - 0.78) < 1e-6


def test_wraparound_overlap_and_range():
    r = disc_ranges(360, [(-3.0, 0.0)], 0.42, 8.0)          # behind: beams around +-180 deg
    assert not np.isnan(r[0]) and not np.isnan(r[359])
    assert np.all(np.isnan(disc_ranges(360, [(0.3, 0.0)], 0.42, 8.0)))   # inside own footprint: skip
    assert np.all(np.isnan(disc_ranges(360, [(20.0, 0.0)], 0.42, 8.0)))  # out of range
    r = disc_ranges(360, [(2.0, 0.0), (1.0, 0.0)], 0.42, 8.0)            # nearest disc wins
    assert abs(r[beam(360, 0)] - 0.58) < 1e-6
