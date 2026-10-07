import math

import numpy as np

from rover_multi.link_model import LinkModel, LinkModelParams


def _truth(x, y, ap=(0.0, 0.0), p0=-38.0, n=3.2):
    return p0 - 10 * n * math.log10(max(math.hypot(x - ap[0], y - ap[1]), 1.0))


def _walk(model, ap=(0.0, 0.0), noise=2.0, seed=0, xs=np.linspace(1, 15, 60)):
    rng = np.random.default_rng(seed)
    for x in xs:
        for y in (-2.0, 0.0, 3.0):
            model.add(x, y, _truth(x, y, ap) + rng.normal(0, noise))


def test_not_ready_before_min_samples():
    m = LinkModel(LinkModelParams(min_samples=20))
    for i in range(19):
        m.add(i * 0.5, 0.0, -50.0)
    m.fit()
    assert m.predict(5.0, 0.0) is None


def test_fits_path_loss_with_known_ap():
    m = LinkModel(LinkModelParams())
    _walk(m)
    m.fit()
    assert abs(m.n - 3.2) < 0.3 and abs(m.p0 + 38.0) < 3.0
    # extrapolate beyond the measured area (a frontier further in)
    assert abs(m.predict(20.0, 0.0) - _truth(20.0, 0.0)) < 3.0


def test_prior_keeps_fit_sane_with_few_close_samples():
    m = LinkModel(LinkModelParams(min_samples=5))
    for x in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        m.add(x, 0.0, _truth(x, 0.0))
    m.fit()
    assert 2.0 <= m.n <= 4.0


def test_estimates_unknown_ap_position():
    ap = (4.0, -3.0)
    m = LinkModel(LinkModelParams(ap_x=0.0, ap_y=0.0, estimate_ap=True, ap_search_step=0.5))
    rng = np.random.default_rng(1)
    for x in np.linspace(-6, 14, 40):
        for y in np.linspace(-8, 6, 8):
            m.add(x, y, _truth(x, y, ap) + rng.normal(0, 2.0))
    m.fit()
    assert math.hypot(m.ap[0] - ap[0], m.ap[1] - ap[1]) <= 1.0


def test_residuals_correct_near_measurements():
    # A wall makes one measured spot 12 dB worse than the model; nearby predictions follow it.
    m = LinkModel(LinkModelParams(residual_radius=2.0))
    _walk(m, noise=0.0)
    for _ in range(30):
        m.add(8.0, 6.0, _truth(8.0, 6.0) - 12.0)
    m.fit()
    near, far = m.predict(8.1, 6.0), m.predict(8.0, -6.0)
    assert near < _truth(8.1, 6.0) - 6.0
    assert abs(far - _truth(8.0, -6.0)) < 3.0


def test_samples_are_binned():
    m = LinkModel(LinkModelParams(bin_size=0.25, min_samples=1))
    for _ in range(1000):
        m.add(1.01, 1.01, -50.0)
    m.fit()
    assert len(m.rssi) == 1 and m.count == 1000
