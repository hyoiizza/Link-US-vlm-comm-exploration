"""RViz markers of the coordinator's goal selection (/team/selection_markers, world frame).

    frontier spheres   green: feasible for some robot, red: link below Q_min for every robot,
                       grey: no path / not evaluated; label "S G Q" (and U of the chosen goals)
    goal arrows        one colour per robot, from the robot to its goal; cube = waiting place
    link samples       measured RSSI per 0.25 m cell, green (>= q_min + 10 dB) .. red (< q_min)
"""
import math

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROBOT_COLORS = [(0.1, 0.45, 1.0), (1.0, 0.55, 0.0), (0.6, 0.2, 0.9), (0.0, 0.7, 0.6)]


def _color(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _marker(frame, stamp, ns, mid, mtype):
    m = Marker()
    m.header.frame_id, m.header.stamp = frame, stamp
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def _fmt(v, spec):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def selection_markers(frame, stamp, robots, tasks, scores, chosen, robot_pos, wait_goals):
    """tasks: {id: Task}; scores: {(robot, id): Score}; chosen: {robot: (id, U)};
    robot_pos: {robot: (x, y)}; wait_goals: {robot: (x, y)}."""
    out = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    out.markers.append(clear)
    mid = 0
    for tid, t in tasks.items():
        s = [scores[(r, tid)] for r in robots if (r, tid) in scores]
        if any(x.feasible for x in s):
            col = _color(0.1, 0.8, 0.2)
        elif s and all(x.reason.startswith('link') for x in s):
            col = _color(0.9, 0.1, 0.1)
        else:
            col = _color(0.5, 0.5, 0.5)
        sphere = _marker(frame, stamp, 'frontiers', mid, Marker.SPHERE)
        sphere.pose.position.x, sphere.pose.position.y = t.x, t.y
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
        sphere.color = col
        out.markers.append(sphere)
        ref = s[0] if s else None
        label = _marker(frame, stamp, 'frontier_labels', mid, Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = t.x, t.y, 0.45
        label.scale.z = 0.18
        label.color = _color(1, 1, 1)
        label.text = (f'S{_fmt(ref.S, ".2f")} G{_fmt(ref.G, ".2f")} Q{_fmt(ref.Q, ".0f")}' if ref else tid)
        out.markers.append(label)
        mid += 1

    for i, robot in enumerate(robots):
        rgb = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        start = robot_pos.get(robot)
        goal = None
        if robot in chosen and chosen[robot][0] in tasks:
            t = tasks[chosen[robot][0]]
            goal = (t.x, t.y)
            text = f'{robot}  U{chosen[robot][1]:.2f}'
        elif robot in wait_goals:
            goal = wait_goals[robot]
            text = f'{robot}  wait'
            cube = _marker(frame, stamp, 'wait', i, Marker.CUBE)
            cube.pose.position.x, cube.pose.position.y = goal
            cube.scale.x = cube.scale.y = cube.scale.z = 0.3
            cube.color = _color(1.0, 0.9, 0.1)
            out.markers.append(cube)
        if goal is None:
            continue
        if start is not None:
            arrow = _marker(frame, stamp, 'assignments', i, Marker.ARROW)
            arrow.points = [Point(x=float(start[0]), y=float(start[1]), z=0.1),
                            Point(x=float(goal[0]), y=float(goal[1]), z=0.1)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.06, 0.15, 0.2
            arrow.color = _color(*rgb)
            out.markers.append(arrow)
        name = _marker(frame, stamp, 'assignment_labels', i, Marker.TEXT_VIEW_FACING)
        name.pose.position.x, name.pose.position.y, name.pose.position.z = goal[0], goal[1], 0.75
        name.scale.z = 0.22
        name.color = _color(*rgb)
        name.text = text
        out.markers.append(name)
    return out


def link_markers(frame, stamp, xy, rssi, q_min, cell=0.25):
    """Measured link cells coloured from red (< q_min) to green (>= q_min + 10 dB)."""
    out = MarkerArray()
    m = _marker(frame, stamp, 'link_samples', 0, Marker.CUBE_LIST)
    m.scale.x = m.scale.y = cell
    m.scale.z = 0.02
    for (x, y), q in zip(xy, rssi):
        m.points.append(Point(x=float(x), y=float(y), z=0.01))
        f = min(max((q - q_min) / 10.0, 0.0), 1.0) if q >= q_min else 0.0
        m.colors.append(_color(1.0 - f, 0.3 + 0.6 * f if q >= q_min else 0.0, 0.0, 0.6))
    out.markers.append(m)
    return out
