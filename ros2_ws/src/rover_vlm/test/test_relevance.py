import math

import numpy as np
import pytest

from rover_vlm.relevance import (
    crop_columns, depth_profile, KeyframeStore, prompt_embedding, relevance, ViewParams)


def _rot_z(yaw):
    """map <- optical: optical Z (forward) along map yaw, X (right) = yaw - 90 deg, Y down."""
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    d = np.array([0.0, 0.0, -1.0])
    return np.c_[r, d, f]


def test_relevance_is_two_prompt_softmax():
    pos, neg = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    img = np.array([[0.8, 0.6], [0.6, 0.8], [1.0, 1.0]])
    S = relevance(img, pos, neg, scale=10.0)
    assert S[0] > 0.5 > S[1] and S[2] == pytest.approx(0.5)
    sp, sn = 0.8, 0.6
    assert S[0] == pytest.approx(math.exp(10 * sp) / (math.exp(10 * sp) + math.exp(10 * sn)))
    # temperature sharpens / flattens
    assert relevance(img[:1], pos, neg, 10.0, temperature=0.5)[0] > S[0]


def test_prompt_ensemble_is_unit_mean():
    e = prompt_embedding([[1.0, 0.0], [0.0, 1.0]])
    assert np.linalg.norm(e) == pytest.approx(1.0) and e[0] == pytest.approx(e[1])


def test_crop_columns():
    assert crop_columns(848, 480, 3) == [(0, 480), (184, 664), (368, 848)]
    assert crop_columns(848, 480, 1) == [(184, 664)]


def _store(**kw):
    s = KeyframeStore(ViewParams(**kw))
    return s


def test_frontier_ahead_gets_the_centered_crop():
    s = _store()
    # camera at origin looking +x; fx = cx = 424 -> 90 deg horizontal FOV
    s.add(0.0, _rot_z(0.0), [0.0, 0.0, 0.2], 424.0, 424.0, crop_columns(848, 480, 3), [0.1, 0.9, 0.3])
    assert s.best_view(3.0, 0.0)[0] == pytest.approx(0.9)            # straight ahead -> center crop
    assert s.best_view(3.0, 2.0)[0] == pytest.approx(0.1)            # to the left -> left crop
    assert s.best_view(3.0, -2.0)[0] == pytest.approx(0.3)           # to the right -> right crop
    assert s.best_view(-3.0, 0.0) is None                            # behind the camera
    assert s.best_view(9.0, 0.0) is None                             # beyond max_range


def test_most_head_on_view_wins_and_occlusion():
    s = _store()
    crops = crop_columns(848, 480, 3)
    s.add(0.0, _rot_z(0.0), [0.0, 0.0, 0.2], 424.0, 424.0, crops, [0.2, 0.2, 0.2])   # sees (3,1) off-center
    s.add(1.0, _rot_z(math.atan2(1, 3)), [0.0, 0.0, 0.2], 424.0, 424.0, crops, [0.7, 0.7, 0.7])
    assert s.best_view(3.0, 1.0)[0] == pytest.approx(0.7)
    # a wall at 1 m in every column of the head-on keyframe hides the frontier there
    s.keyframes[1].depth = np.full(848, 1.0)
    assert s.best_view(3.0, 1.0)[0] == pytest.approx(0.2)


def test_keyframe_policy():
    s = _store(min_translation=0.5, min_rotation_deg=15)
    assert s.needs_keyframe(_rot_z(0), [0, 0, 0])
    s.add(0.0, _rot_z(0), [0, 0, 0], 424, 424, [(0, 480)], [0.5])
    assert not s.needs_keyframe(_rot_z(math.radians(10)), [0.2, 0, 0])
    assert s.needs_keyframe(_rot_z(math.radians(20)), [0, 0, 0])
    assert s.needs_keyframe(_rot_z(0), [0.6, 0, 0])


def test_depth_profile_ignores_invalid():
    d = np.zeros((100, 4))
    d[40:60, 1] = 2.0
    d[40:60, 2] = np.nan
    p = depth_profile(d)
    assert np.isnan(p[0]) and p[1] == pytest.approx(2.0) and np.isnan(p[2])


def test_percentile_spreads_narrow_scores():
    store = KeyframeStore()
    R = np.eye(3)
    for k, s in enumerate(np.linspace(0.44, 0.52, 10)):
        store.add(float(k), R, np.array([k * 1.0, 0.0, 0.3]), 400.0, 320.0, [(0, 480)], [s])
    assert store.percentile(0.40) == 0.0
    assert store.percentile(0.60) == 1.0
    assert abs(store.percentile(0.48) - 0.5) < 0.06      # middle of the band -> ~0.5
    assert store.percentile(0.52) > 0.9
    assert KeyframeStore().percentile(0.5) is None
