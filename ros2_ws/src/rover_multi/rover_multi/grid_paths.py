"""Path lengths from the robot to many goals with one Dijkstra over the global costmap.

Free of ROS so it can be unit tested. Replaces one Nav2 ComputePathToPose per frontier in
the bids: those calls shared the planner_server with the robot's own navigation and made
its path requests time out (goals aborted right after they were sent).

The grid is the Nav2 costmap as published (nav_msgs/OccupancyGrid: 0-100, 99 inscribed,
100 lethal, -1 unknown). Like the planner (allow_unknown), unknown cells are traversable;
cells at or above `blocked_cost` are not, since the 2D planner checks only the robot centre.
"""
import heapq
import math
import warnings

import numpy as np

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # Jetson scipy 1.8 warns about numpy 1.26
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
except Exception:                            # e.g. robot2: numpy 2 (ultralytics) + system scipy 1.8
    coo_matrix = dijkstra = None


def dijkstra_grid(free, start, step):
    """(dist [H*W], pred [H*W]) over the 8-connected free cells, pure Python (no scipy)."""
    h, w = free.shape
    n = h * w
    dist = np.full(n, np.inf)
    pred = np.full(n, -9999, dtype=np.int64)
    flat = free.ravel()
    s = start[0] * w + start[1]
    dist[s] = 0.0
    nbrs = [(di, dj, step * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    heap = [(0.0, s)]
    done = np.zeros(n, dtype=bool)
    while heap:
        d, k = heapq.heappop(heap)
        if done[k]:
            continue
        done[k] = True
        i, j = divmod(k, w)
        for di, dj, c in nbrs:
            ii, jj = i + di, j + dj
            if 0 <= ii < h and 0 <= jj < w:
                kk = ii * w + jj
                if flat[kk] and not done[kk] and d + c < dist[kk]:
                    dist[kk] = d + c
                    pred[kk] = k
                    heapq.heappush(heap, (d + c, kk))
    return dist, pred


def downsample_min(cost, factor):
    """Min-pool the cost grid by an integer factor: a coarse cell is passable if any part is.

    Max-pooling closed doorways (paths ~2x the Nav2 plan on the 2026-10-01 map). Min-pooling
    cannot open a wall: inflated obstacles are >= 2 x inscribed radius (0.6 m) thick, far
    more than one coarse cell."""
    if factor <= 1:
        return cost
    h, w = cost.shape
    H, W = -(-h // factor), -(-w // factor)
    padded = np.full((H * factor, W * factor), 100, dtype=cost.dtype)
    padded[:h, :w] = cost
    free = np.where(padded < 0, 0, padded)          # unknown is traversable like free
    return free.reshape(H, factor, W, factor).min(axis=(1, 3))


class GridPaths:
    """Shortest paths from `start` over one costmap snapshot.

    cost: [H, W] int array, row 0 at origin y; resolution [m/cell]; origin (x, y) of cell (0, 0).
    """

    def __init__(self, cost, resolution, origin, start, blocked_cost=99, downsample=2,
                 escape_radius=0.4):
        cost = downsample_min(np.asarray(cost, dtype=np.int16), downsample)
        self.res = resolution * max(downsample, 1)
        self.origin = origin
        self.shape = cost.shape
        h, w = cost.shape
        free = cost < blocked_cost
        # The robot may already sit in inflated space (e.g. next to a wall): let it out.
        si, sj = self.cell(*start)
        self.start_cell = (min(max(si, 0), h - 1), min(max(sj, 0), w - 1))
        r = int(math.ceil(escape_radius / self.res))
        i0, j0 = self.start_cell
        ii, jj = np.ogrid[:h, :w]
        free |= (ii - i0) ** 2 + (jj - j0) ** 2 <= r * r

        if dijkstra is None:
            self.dist, self.pred = dijkstra_grid(free, self.start_cell, self.res)
            self.dist = self.dist.reshape(h, w)
            return
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, weights = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):   # each undirected neighbour pair once
            ja, jb = max(0, -dj), w - max(0, dj)                # columns j with j + dj inside
            a = (slice(0, h - di), slice(ja, jb))
            b = (slice(di, h), slice(ja + dj, jb + dj))
            both = free[a] & free[b]
            rows.append(idx[a][both])
            cols.append(idx[b][both])
            weights.append(np.full(int(both.sum()), self.res * math.hypot(di, dj)))
        graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(h * w, h * w)).tocsr()
        start_idx = i0 * w + j0
        self.dist, self.pred = dijkstra(graph, directed=False, indices=start_idx, return_predecessors=True)
        self.dist = self.dist.reshape(h, w)

    def cell(self, x, y):
        return (int(math.floor((y - self.origin[1]) / self.res)),
                int(math.floor((x - self.origin[0]) / self.res)))

    def center(self, i, j):
        return (self.origin[0] + (j + 0.5) * self.res, self.origin[1] + (i + 0.5) * self.res)

    def path_to(self, x, y, goal_radius=0.3):
        """(length [m], [(x, y)] from start to goal) or None when unreachable.

        The goal counts as reached anywhere within goal_radius (frontier goals often sit in
        unknown or inflated cells); the length includes the last straight bit to the goal."""
        h, w = self.shape
        gi, gj = self.cell(x, y)
        r = int(math.ceil(goal_radius / self.res))
        i0, i1 = max(gi - r, 0), min(gi + r + 1, h)
        j0, j1 = max(gj - r, 0), min(gj + r + 1, w)
        if i0 >= i1 or j0 >= j1:
            return None
        best = None
        for i in range(i0, i1):
            for j in range(j0, j1):
                d = self.dist[i, j]
                if not np.isfinite(d):
                    continue
                cx, cy = self.center(i, j)
                gap = math.hypot(cx - x, cy - y)
                if gap > goal_radius:
                    continue
                if best is None or d + gap < best[0]:
                    best = (d + gap, i, j)
        if best is None:
            return None
        length, i, j = best
        cells = []
        k = i * w + j
        while k >= 0:
            cells.append(divmod(int(k), w))
            k = self.pred[k]
        pts = [self.center(ci, cj) for ci, cj in reversed(cells)]
        return float(length), pts + [(x, y)]
