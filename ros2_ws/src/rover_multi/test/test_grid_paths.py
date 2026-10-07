import math

import numpy as np

from rover_multi.grid_paths import downsample_min, GridPaths


def test_straight_and_around_wall():
    cost = np.zeros((40, 40), dtype=np.int16)            # 2 x 2 m at 0.05
    g = GridPaths(cost, 0.05, (0.0, 0.0), (0.2, 1.0), downsample=1)
    length, pts = g.path_to(1.8, 1.0)
    assert abs(length - 1.6) < 0.1
    cost[5:40, 20] = 100                                 # wall at x = 1.0, gap at the bottom
    g = GridPaths(cost, 0.05, (0.0, 0.0), (0.2, 1.0), downsample=1)
    length, pts = g.path_to(1.8, 1.0)
    assert 2.1 < length < 2.6                            # detour via the gap: ~2 x hypot(0.8, 0.8)
    assert min(p[1] for p in pts) < 0.3


def test_unreachable_and_unknown():
    cost = np.full((40, 40), -1, dtype=np.int16)          # unknown is traversable
    g = GridPaths(cost, 0.05, (0.0, 0.0), (0.2, 0.2), downsample=2)
    assert g.path_to(1.8, 1.8) is not None
    cost[:, 14:26] = 100                                 # closed wall, inflated (0.6 m) like the costmap
    g = GridPaths(cost, 0.05, (0.0, 0.0), (0.2, 0.2), downsample=2)
    assert g.path_to(1.8, 1.8) is None


def test_escape_from_inflation():
    cost = np.full((40, 40), 99, dtype=np.int16)
    cost[:, 12:] = 0                                     # robot inside inflated band at x = 0.3
    g = GridPaths(cost, 0.05, (0.0, 0.0), (0.3, 1.0), downsample=1, escape_radius=0.4)
    assert g.path_to(1.5, 1.0) is not None


def test_downsample_min_keeps_doorway():
    c = np.array([[0, 5, 1], [2, 100, -1]], dtype=np.int16)
    assert downsample_min(c, 2).tolist() == [[0, 0]]
    cost = np.full((40, 40), 100, dtype=np.int16)
    cost[:, :18] = 0
    cost[:, 23:] = 0
    cost[19:21, 18:23] = 0                              # 0.1 m doorway through a 0.25 m wall
    g = GridPaths(cost, 0.05, (0.0, 0.0), (0.3, 1.0), downsample=2)
    assert g.path_to(1.7, 1.0) is not None


def test_pure_python_fallback_matches_scipy():
    import rover_multi.grid_paths as gp
    rng = np.random.default_rng(1)
    cost = np.zeros((60, 60), dtype=np.int16)
    cost[rng.random((60, 60)) < 0.15] = 100
    cost[25:35, 25:35] = 0
    goals = [(2.5, 2.5), (0.3, 2.8), (2.9, 0.2)]
    ref = gp.GridPaths(cost, 0.05, (0.0, 0.0), (1.5, 1.5), downsample=1)
    saved = gp.dijkstra
    gp.dijkstra = None
    try:
        alt = gp.GridPaths(cost, 0.05, (0.0, 0.0), (1.5, 1.5), downsample=1)
    finally:
        gp.dijkstra = saved
    for x, y in goals:
        a, b = ref.path_to(x, y), alt.path_to(x, y)
        assert (a is None) == (b is None)
        if a:
            assert abs(a[0] - b[0]) < 1e-6
