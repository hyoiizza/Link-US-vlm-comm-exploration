"""Semantic re-weighting of frontier gains, free of ROS so it can be unit tested.

The segmentation model separates floor and wall; every other observed class is
treated as an "object" (furniture, debris, possibly a person). Around a frontier
goal (disc of `radius` m) the semantic grid gives the share of known cells per kind:

    floor  -> open space continues (room / corridor): more promising
    wall   -> frontier hugging a wall, often a sensor-noise sliver: less promising
    object -> something that is neither floor nor wall: worth a look when searching for people

    factor = 1 + floor_weight * floor - wall_weight * wall + object_weight * object

clamped to [min_factor, max_factor]. A frontier with fewer than `min_known_cells`
confident cells around it keeps factor 1 (no evidence either way). All weights 0
turns the re-weighting off.
"""
from dataclasses import dataclass

import numpy as np

UNKNOWN = 255


@dataclass(frozen=True)
class SemanticWeights:
    floor: float = 0.5
    wall: float = 0.5
    object: float = 0.5
    radius: float = 1.0
    min_known_cells: int = 20
    min_confidence: int = 50       # [0-100] share of votes a cell's class needs to count
    min_factor: float = 0.2
    max_factor: float = 2.0


@dataclass(frozen=True)
class Shares:
    floor: float = 0.0
    wall: float = 0.0
    object: float = 0.0
    known: int = 0                 # confident cells inside the disc


class SemanticLayer:
    """Class labels of one rover_msgs/SemanticGrid, in its own frame (no rotation)."""

    def __init__(self, labels, confidence, resolution, origin_x, origin_y, class_names):
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.confidence = np.asarray(confidence, dtype=np.uint8)
        self.res = float(resolution)
        self.x0, self.y0 = float(origin_x), float(origin_y)
        names = list(class_names)
        self.floor = names.index('floor') if 'floor' in names else None
        self.wall = names.index('wall') if 'wall' in names else None

    def shares(self, x, y, radius, min_confidence):
        h, w = self.labels.shape
        if h == 0 or w == 0:
            return Shares()
        ix0 = max(int(np.floor((x - radius - self.x0) / self.res)), 0)
        ix1 = min(int(np.floor((x + radius - self.x0) / self.res)) + 1, w)
        iy0 = max(int(np.floor((y - radius - self.y0) / self.res)), 0)
        iy1 = min(int(np.floor((y + radius - self.y0) / self.res)) + 1, h)
        if ix0 >= ix1 or iy0 >= iy1:
            return Shares()
        labels = self.labels[iy0:iy1, ix0:ix1]
        conf = self.confidence[iy0:iy1, ix0:ix1]
        cy, cx = np.mgrid[iy0:iy1, ix0:ix1]
        disc = np.hypot(self.x0 + (cx + 0.5) * self.res - x, self.y0 + (cy + 0.5) * self.res - y) <= radius
        known = disc & (labels != UNKNOWN) & (conf >= min_confidence)
        n = int(known.sum())
        if n == 0:
            return Shares()
        floor = int((known & (labels == self.floor)).sum()) if self.floor is not None else 0
        wall = int((known & (labels == self.wall)).sum()) if self.wall is not None else 0
        return Shares(floor / n, wall / n, (n - floor - wall) / n, n)


def gain_factor(shares, weights):
    if shares.known < weights.min_known_cells:
        return 1.0
    factor = 1.0 + weights.floor * shares.floor - weights.wall * shares.wall + weights.object * shares.object
    return float(min(max(factor, weights.min_factor), weights.max_factor))
