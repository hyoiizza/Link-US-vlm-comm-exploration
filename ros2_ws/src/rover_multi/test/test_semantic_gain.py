import numpy as np

from rover_multi.semantic_gain import gain_factor, SemanticLayer, SemanticWeights, Shares, UNKNOWN

NAMES = ['background', 'floor', 'wall', 'stairs']
BG, FLOOR, WALL = 0, 1, 2


def _layer(labels, conf=100):
    labels = np.asarray(labels, dtype=np.uint8)
    return SemanticLayer(labels, np.full(labels.shape, conf, np.uint8), 0.1, -1.0, -1.0, NAMES)


def test_shares_count_only_the_disc():
    labels = np.full((20, 20), FLOOR)
    labels[:, 15:] = WALL                     # x >= 0.5 m
    s = _layer(labels).shares(0.0, 0.0, 0.3, 50)
    assert s.known > 0 and s.floor == 1.0 and s.wall == 0.0


def test_unknown_and_unconfident_cells_are_ignored():
    labels = np.full((20, 20), UNKNOWN)
    labels[10, 10] = WALL
    assert _layer(labels).shares(0.0, 0.0, 0.5, 50).known == 1
    assert _layer(labels, conf=40).shares(0.0, 0.0, 0.5, 50).known == 0


def test_other_classes_count_as_objects():
    labels = np.full((20, 20), BG)
    labels[:10] = FLOOR
    s = _layer(labels).shares(0.0, 0.0, 0.5, 50)
    assert abs(s.floor + s.object - 1.0) < 1e-9 and s.object > 0.3


def test_goal_outside_the_grid():
    assert _layer(np.full((5, 5), FLOOR)).shares(10.0, 10.0, 1.0, 50).known == 0


def test_factor():
    w = SemanticWeights(floor=0.5, wall=0.5, object=0.5, min_known_cells=10)
    assert gain_factor(Shares(1.0, 0.0, 0.0, 100), w) == 1.5
    assert gain_factor(Shares(0.0, 1.0, 0.0, 100), w) == 0.5
    assert gain_factor(Shares(0.0, 1.0, 0.0, 5), w) == 1.0            # too little evidence
    assert gain_factor(Shares(0.0, 1.0, 0.0, 100), SemanticWeights(0.0, 0.0, 0.0)) == 1.0
    assert gain_factor(Shares(0.0, 1.0, 0.0, 100), SemanticWeights(wall=5.0)) == 0.2   # clamped
