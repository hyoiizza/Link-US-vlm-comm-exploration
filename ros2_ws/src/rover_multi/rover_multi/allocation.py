"""Task allocation logic of the coordinator, free of ROS so it can be unit tested.

* merge_candidates: frontiers reported by several robots -> one de-duplicated task list
* ssi_assign:       sequential single-item auction over the robots' bids, with
                    "spread out" penalties and a switch threshold for busy robots
"""
from dataclasses import dataclass, field
import math


@dataclass
class Task:
    id: str
    x: float
    y: float
    gain: float
    sources: list = field(default_factory=list)   # robots that reported it
    region_id: str = ''
    label: str = ''
    reports: list = field(default_factory=list)   # extra per-robot data of the merged candidates


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def merge_candidates(candidates, merge_radius, id_prefix='t', start_seq=0):
    """Merge candidates [(robot, x, y, gain, region_id, label[, extra]), ...] closer than merge_radius.

    Higher-gain candidates are kept as representatives, so a merged task keeps
    the best reported goal position. Every merged candidate's optional `extra`
    is kept in Task.reports as (robot, extra). Returns (tasks, next_seq).
    """
    tasks = []
    seq = start_seq
    for c in sorted(candidates, key=lambda c: -c[3]):
        robot, x, y, gain, region_id, label = c[:6]
        extra = c[6] if len(c) > 6 else None
        for task in tasks:
            if distance((task.x, task.y), (x, y)) <= merge_radius:
                if robot not in task.sources:
                    task.sources.append(robot)
                break
        else:
            task = Task(f'{id_prefix}{seq}', x, y, gain, [robot], region_id, label)
            tasks.append(task)
            seq += 1
        if extra is not None:
            task.reports.append((robot, extra))
    return tasks, seq


def _reduce(utility, fraction):
    """Lower a utility by `fraction` of its magnitude (works for negative utilities too)."""
    return utility - fraction * abs(utility)


def ssi_assign(robots, tasks, utilities, near_radius, near_penalty,
               current=None, switch_threshold=0.0):
    """Sequential single-item auction.

    robots:     robots that take part
    tasks:      [Task]
    utilities:  {(robot, task_id): utility} from the bids (missing = robot did not bid)
    current:    {robot: (task_id, utility)} for robots already driving to a task that is
                still open; any other task needs to beat it by switch_threshold x |utility|
    Each round awards the single best (robot, task) pair; tasks near an awarded task
    are penalised for the remaining robots so the team spreads out.

    Returns {robot: (task_id, utility)} for every robot that got a task.
    """
    current = current or {}
    by_id = {t.id: t for t in tasks}
    util = dict(utilities)

    def effective(robot, task_id, value):
        cur = current.get(robot)
        if cur is None or cur[0] == task_id:
            return value
        return value - switch_threshold * abs(cur[1])

    open_robots = [r for r in robots]
    open_tasks = set(by_id)
    awards = {}
    while open_robots and open_tasks:
        best = None
        for robot in open_robots:
            for task_id in open_tasks:
                value = util.get((robot, task_id))
                if value is None:
                    continue
                score = effective(robot, task_id, value)
                if best is None or score > best[0]:
                    best = (score, robot, task_id, value)
        if best is None:
            break
        _, robot, task_id, value = best
        awards[robot] = (task_id, value)
        open_robots.remove(robot)
        open_tasks.discard(task_id)

        won = by_id[task_id]
        for other_id in open_tasks:
            if distance((won.x, won.y), (by_id[other_id].x, by_id[other_id].y)) <= near_radius:
                for other_robot in open_robots:
                    key = (other_robot, other_id)
                    if key in util:
                        util[key] = _reduce(util[key], near_penalty)
    return awards
