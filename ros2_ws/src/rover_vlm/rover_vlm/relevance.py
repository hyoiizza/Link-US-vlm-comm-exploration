"""Frontier semantic relevance from keyframe crops, free of ROS so it can be unit tested.

Relevance of one image crop I (paper, section B):

    s+ = sim(E_I(I), E_T(T+)),   s- = sim(E_I(I), E_T(T-))
    S  = exp(k s+) / (exp(k s+) + exp(k s-)) = sigmoid(k (s+ - s-))

sim is the cosine of the L2-normalized embeddings and k the scale: the model's
own logit scale (similarity "scaled") or 1 (similarity "cosine"), divided by an
optional temperature. The per-model bias cancels in the two-prompt softmax.
With several prompts per side, their embeddings are averaged and renormalized.

Keyframes: the color image is cut into n square crops across its width; each
crop's S is computed once when the keyframe is taken. A frontier gets the S of
the crop that sees it most head-on (smallest offset from the crop center,
relative to the crop half width; then the closest keyframe; then the newest), among
crops where it lies in front of the camera within [min_range, max_range] and is
not hidden behind nearer depth. Only images actually captured are used: a
frontier no keyframe has seen stays unobserved.

Normalization: raw S of SigLIP2 on indoor scenes stays in a narrow band (0.41-0.64
on 2026-10-01), too small to matter next to the gain and cost terms. A frontier's
S can be reported as its percentile among every crop the robot has scored so far,
F(s) = (#{s_k < s} + 0.5 #{s_k = s}) / n, which spreads it over [0, 1] and puts
the average view at 0.5 (the value unobserved frontiers get).
"""
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field
import math
import warnings

import numpy as np


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def prompt_embedding(embeds):
    """One unit vector for a prompt set: mean of the (unit) prompt embeddings, renormalized."""
    return normalize_rows(normalize_rows(np.atleast_2d(embeds)).mean(axis=0))


def relevance(image_embeds, pos_embed, neg_embed, scale=1.0, temperature=1.0):
    """S for each image embedding (rows); pos/neg are unit prompt embeddings."""
    img = normalize_rows(np.atleast_2d(image_embeds))
    k = scale / temperature
    return 1.0 / (1.0 + np.exp(-k * (img @ pos_embed - img @ neg_embed)))


def crop_columns(width, height, n):
    """[(u0, u1)] of n square (height x height) crops spread evenly across the image width."""
    side = min(width, height)
    if n == 1:
        starts = [(width - side) // 2]
    else:
        starts = [round(i * (width - side) / (n - 1)) for i in range(n)]
    return [(s, s + side) for s in starts]


def depth_profile(depth_m, band=0.2):
    """Per image column median of the valid depth [m] in the middle `band` of rows (nan when none)."""
    h = depth_m.shape[0]
    rows = depth_m[int(h * (0.5 - band / 2)):int(h * (0.5 + band / 2)) + 1].astype(np.float64)
    rows[~np.isfinite(rows) | (rows <= 0)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-nan columns
        return np.nanmedian(rows, axis=0)


@dataclass
class Keyframe:
    id: int
    stamp: float
    R: np.ndarray                 # 3x3 rotation map <- camera optical frame
    t: np.ndarray                 # camera position in map
    fx: float
    cx: float
    crops: list                   # [(u0, u1)]
    scores: np.ndarray            # S per crop
    depth: np.ndarray = None      # depth_profile per column [m], or None

    def to_camera(self, x, y):
        """Frontier (x, y) at camera height in the optical frame (X right, Z forward)."""
        p = np.array([x, y, self.t[2]]) - self.t
        return self.R.T @ p


@dataclass
class ViewParams:
    min_range: float = 0.5        # [m] along the optical axis
    max_range: float = 6.0
    occlusion_margin: float = 0.5 # [m] depth nearer than the frontier by this much hides it
    min_translation: float = 0.5  # [m] new keyframe after moving this far ...
    min_rotation_deg: float = 15.0   # ... or turning this much
    max_keyframes: int = 500


@dataclass
class KeyframeStore:
    params: ViewParams = field(default_factory=ViewParams)
    keyframes: list = field(default_factory=list)
    next_id: int = 0
    score_sum: float = 0.0        # over every crop ever scored (not only the kept keyframes)
    score_count: int = 0
    sorted_scores: list = field(default_factory=list)   # every crop S ever scored, ascending

    def needs_keyframe(self, R, t):
        if not self.keyframes:
            return True
        last = self.keyframes[-1]
        moved = np.hypot(*(np.asarray(t)[:2] - last.t[:2])) >= self.params.min_translation
        # angle between the optical axes projected on the floor
        a, b = last.R[:2, 2], np.asarray(R)[:2, 2]
        turn = math.degrees(abs(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)))
        return moved or turn >= self.params.min_rotation_deg

    def add(self, stamp, R, t, fx, cx, crops, scores, depth=None):
        kf = Keyframe(self.next_id, stamp, np.asarray(R, float), np.asarray(t, float), fx, cx,
                      list(crops), np.asarray(scores, float), depth)
        self.next_id += 1
        self.score_sum += float(kf.scores.sum())
        self.score_count += len(kf.scores)
        for s in kf.scores:
            insort(self.sorted_scores, float(s))
        self.keyframes.append(kf)
        if len(self.keyframes) > self.params.max_keyframes:
            self.keyframes.pop(0)
        return kf

    def mean_score(self):
        """Mean S over every crop scored so far (the robot's average view), None before the first."""
        return self.score_sum / self.score_count if self.score_count else None

    def percentile(self, s):
        """Empirical CDF of s among all crops scored so far (ties count half), None before the first."""
        n = len(self.sorted_scores)
        if n == 0:
            return None
        lo, hi = bisect_left(self.sorted_scores, s), bisect_right(self.sorted_scores, s)
        return (lo + 0.5 * (hi - lo)) / n

    def best_view(self, x, y):
        """(S, keyframe id, crop index) of the view that sees frontier (x, y) best, or None."""
        p = self.params
        best = None
        for kf in self.keyframes:
            X, _, Z = kf.to_camera(x, y)
            if not p.min_range <= Z <= p.max_range:
                continue
            u = kf.cx + kf.fx * X / Z
            if kf.depth is not None:
                col = int(round(u))
                if 0 <= col < len(kf.depth) and np.isfinite(kf.depth[col]) \
                        and kf.depth[col] < Z - p.occlusion_margin:
                    continue   # something nearer is in the way
            for i, (u0, u1) in enumerate(kf.crops):
                if not u0 <= u < u1:
                    continue
                offset = abs(u - (u0 + u1) / 2) / ((u1 - u0) / 2)
                key = (offset, Z, -kf.stamp)
                if best is None or key < best[0]:
                    best = (key, float(kf.scores[i]), kf.id, i)
        return None if best is None else best[1:]
