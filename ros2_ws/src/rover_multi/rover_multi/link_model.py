"""Wi-Fi link quality model for goal selection, free of ROS so it can be unit tested.

Predicts the RSSI [dBm] at a world position from the robots' own measurements with
a log-distance path-loss model around the access point (AP):

    RSSI(d) = P0 - 10 * n * log10(max(d, d0) / d0)          (d = distance to the AP)

P0 and n are fitted by least squares to the samples, pulled towards a prior
(prior_weight pseudo-samples) so the fit stays sane while all samples are still
close to the start. Near measured places the prediction is corrected by the
measured residuals (inverse-distance weighting within residual_radius), so walls
the model does not know about still show up where the robots have been.

The AP position does not have to be known: with estimate_ap the fit also searches
the AP position (grid search around the prior guess, penalised by its distance to
the guess). That only works once the samples are spread out, so the guess (e.g.
the start / base station) matters early on.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LinkModelParams:
    ap_x: float = 0.0             # AP position in world (or the guess when estimate_ap)
    ap_y: float = 0.0
    estimate_ap: bool = False
    ap_search_radius: float = 10.0   # [m] around the guess
    ap_search_step: float = 0.5      # [m]
    ap_prior_sigma: float = 5.0      # [m] how far from the guess the AP plausibly is
    d0: float = 1.0               # reference distance [m]
    p0_prior: float = -40.0       # RSSI at d0 [dBm]
    n_prior: float = 3.0          # path-loss exponent (2 free space, 3-4 indoor)
    prior_weight: float = 20.0    # the prior counts as this many samples
    n_bounds: tuple = (1.5, 6.0)
    residual_radius: float = 2.0  # [m] measured residuals correct predictions this close
    min_samples: int = 20         # no prediction before this many samples
    bin_size: float = 0.25        # [m] samples are averaged per cell (bounded size, less noise)


class LinkModel:
    def __init__(self, params=LinkModelParams()):
        self.p = params
        self.bins = {}                # cell -> [sum x, sum y, sum rssi, count]
        self.count = 0
        self.xy = np.zeros((0, 2))
        self.rssi = np.zeros(0)
        self.p0, self.n = params.p0_prior, params.n_prior
        self.ap = (params.ap_x, params.ap_y)
        self.residuals = np.zeros(0)

    @property
    def ready(self):
        return self.count >= self.p.min_samples

    def add(self, x, y, rssi):
        key = (math.floor(x / self.p.bin_size), math.floor(y / self.p.bin_size))
        b = self.bins.setdefault(key, [0.0, 0.0, 0.0, 0])
        b[0] += x
        b[1] += y
        b[2] += rssi
        b[3] += 1
        self.count += 1

    def _collect(self):
        v = np.array(list(self.bins.values())) if self.bins else np.zeros((0, 4))
        self.xy = v[:, :2] / v[:, 3:4] if len(v) else np.zeros((0, 2))
        self.rssi = v[:, 2] / v[:, 3] if len(v) else np.zeros(0)

    def _log_d(self, ap, xy):
        d = np.maximum(np.hypot(xy[:, 0] - ap[0], xy[:, 1] - ap[1]), self.p.d0)
        return np.log10(d / self.p.d0)

    def _fit_at(self, ap):
        """Regularized least squares for (P0, n) with the AP at `ap`; returns (p0, n, sse)."""
        L = self._log_d(ap, self.xy)
        w = self.p.prior_weight
        # rows: samples  rssi = P0 - 10 n L ; prior  P0 = p0_prior, n = n_prior (each weight w)
        A = np.vstack([np.c_[np.ones_like(L), -10.0 * L],
                       [math.sqrt(w), 0.0], [0.0, math.sqrt(w) * 10.0]])
        b = np.concatenate([self.rssi, [math.sqrt(w) * self.p.p0_prior, math.sqrt(w) * 10.0 * self.p.n_prior]])
        (p0, n), *_ = np.linalg.lstsq(A, b, rcond=None)
        n = min(max(n, self.p.n_bounds[0]), self.p.n_bounds[1])
        sse = float(np.sum((self.rssi - (p0 - 10.0 * n * L)) ** 2))
        return p0, n, sse

    def fit(self):
        self._collect()
        if len(self.rssi) == 0:
            return
        if self.p.estimate_ap:
            r, s = self.p.ap_search_radius, self.p.ap_search_step
            best = None
            for dx in np.arange(-r, r + 1e-9, s):
                for dy in np.arange(-r, r + 1e-9, s):
                    ap = (self.p.ap_x + dx, self.p.ap_y + dy)
                    p0, n, sse = self._fit_at(ap)
                    # position prior: sse is in dBm^2, scale the penalty by the sample noise level
                    cost = sse + len(self.rssi) * 4.0 * (dx * dx + dy * dy) / self.p.ap_prior_sigma ** 2
                    if best is None or cost < best[0]:
                        best = (cost, ap, p0, n)
            _, self.ap, self.p0, self.n = best
        else:
            self.p0, self.n, _ = self._fit_at(self.ap)
        self.residuals = self.rssi - self.model(self.xy)

    def model(self, xy):
        return self.p0 - 10.0 * self.n * self._log_d(self.ap, np.atleast_2d(xy))

    def predict(self, x, y):
        """Predicted RSSI [dBm] at (x, y), or None before min_samples."""
        if not self.ready:
            return None
        q = float(self.model([x, y])[0])
        if len(self.residuals) == len(self.rssi) and len(self.rssi):
            d = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
            near = d < self.p.residual_radius
            if near.any():
                w = 1.0 / np.maximum(d[near], 0.1) ** 2
                # fade the correction out towards residual_radius
                fade = float(np.clip(1.0 - d[near].min() / self.p.residual_radius, 0.0, 1.0))
                q += fade * float(np.sum(w * self.residuals[near]) / np.sum(w))
        return q
