"""Upper layer of the team: collects candidates, runs auctions, awards tasks.

Runs once for the team, on the leader robot (robot1). Robots that lose it
switch to standalone mode on their own (see team_agent).

Auctions are started when
  * a robot has no task (just joined, reached / failed its task),
  * a robot was lost (its task goes back to the pool),
  * periodically (auction.reevaluate_period_s) to pick up new frontiers;
    busy robots only switch if the new task beats theirs by switch_threshold.

selection.method picks how the bids become goals:
  * auction: sequential single-item auction on the robots' own utilities (ssi_assign)
  * frontier / comm_aware / semantic_only / proposed: joint goal selection of
    goal_selection.py (the paper's method and its baselines). The robots only report
    Nav2 path lengths; the coordinator scores and assigns. There is no periodic
    re-evaluation: a new selection runs when a robot reaches / fails its goal, is idle
    or lost, or when its goal's frontier has vanished from every robot's candidates.
"""

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from rover_msgs.msg import (
    Assignment, AuctionAnnouncement, Award, Bid, Heartbeat, LinkQuality, TaskCandidate, TaskCandidates,
    TaskStatus)
from rover_multi import team_interface as team
from rover_multi.allocation import distance, merge_candidates, ssi_assign, _reduce
from rover_multi.goal_selection import merge_reports, METHODS, Option, select_goals, SelectionParams
from rover_multi.link_model import LinkModel, LinkModelParams
from rover_multi.team_viz import link_markers, selection_markers
from visualization_msgs.msg import MarkerArray

WAIT_LABEL = 'wait'   # label of the "wait where the link is good" task

BUSY = (TaskStatus.ACCEPTED, TaskStatus.ACTIVE)


class TeamCoordinator(Node):
    def __init__(self):
        super().__init__('team_coordinator')
        p = self._declare
        self.world_frame = p('world_frame', 'world')
        self.robots = list(p('robots', ['robot1', 'robot2']))
        self.host_robot = p('host_robot', 'robot1')
        self.hb_period = p('heartbeat.period_s', 1.0)
        self.hb_timeout = p('heartbeat.timeout_s', 4.0)
        self.merge_radius = p('candidates.merge_radius_m', 1.0)
        self.max_age = p('candidates.max_age_s', 10.0)
        self.completed_hold = p('candidates.completed_hold_s', 5.0)
        self.bid_window = p('auction.bid_window_s', 2.0)
        self.reevaluate_period = p('auction.reevaluate_period_s', 8.0)
        self.switch_threshold = p('auction.switch_threshold', 0.25)
        self.near_radius = p('ssi.near_assigned_radius_m', 3.0)
        self.near_penalty = p('ssi.near_assigned_penalty', 0.5)
        self.fail_radius = p('failure.penalty_radius_m', 1.0)
        self.fail_penalty = p('failure.utility_penalty', 0.3)
        self.max_failures = p('failure.max_failures', 2)
        self.blacklist_duration = p('failure.blacklist_duration_s', 60.0)
        self.method = p('selection.method', 'auction')
        if self.method != 'auction' and self.method not in METHODS:
            raise ValueError(f'selection.method {self.method!r}: expected auction or one of {METHODS}')
        self.joint = self.method != 'auction'
        self.q_min = p('selection.q_min', -70.0)
        if self.joint:
            self.selection = SelectionParams(
                method=self.method,
                alpha=p('selection.alpha', 0.5),
                lam=p('selection.lambda', 1.0),
                g_ref=p('selection.g_ref', 700.0),
                d_ref=p('selection.d_ref', 25.0),
                q_min=self.q_min,
                semantic_default=p('selection.semantic_default', 0.5),
                unobserved_semantic=p('selection.unobserved_semantic', 'mean_view'))
        self.vanish_timeout = p('selection.goal_vanished_s', 4.0)
        self.keep_bonus = p('selection.keep_goal_bonus', 0.0)
        joint_bid_window = p('selection.bid_window_s', 6.0)
        if self.joint:
            self.bid_window = joint_bid_window

        # Link model: predicted RSSI at the frontiers (Q), fitted to the robots' measurements.
        self.link_model = LinkModel(LinkModelParams(
            ap_x=p('link.ap_x', 0.0), ap_y=p('link.ap_y', 0.0),
            estimate_ap=p('link.estimate_ap', False),
            ap_search_radius=p('link.ap_search_radius_m', 10.0),
            ap_prior_sigma=p('link.ap_prior_sigma_m', 5.0),
            p0_prior=p('link.p0_prior_dbm', -40.0), n_prior=p('link.n_prior', 3.0),
            prior_weight=p('link.prior_weight', 20.0),
            residual_radius=p('link.residual_radius_m', 2.0),
            min_samples=p('link.min_samples', 20)))
        self.wait_margin = p('link.wait_margin_db', 3.0)
        self.link_drop_s = p('link.drop_reselect_s', 3.0)
        self.link_latest = {}        # robot -> (rclpy Time, rssi, connected)
        self.link_bad_since = {}     # robot -> rclpy Time its measured link fell below q_min
        self.robot_pos = {}          # robot -> (x, y) in world from its heartbeat
        self.link_fits = 0

        self.last_seen = {}          # robot -> rclpy Time of last heartbeat
        self.candidates = {}         # robot -> (rclpy Time, [TaskCandidate])
        self.assignments = {}        # robot -> dict(task=TaskCandidate, utility, status, auction_id)
        self.failures = []           # dict(x, y, count, time)
        self.completed = []          # (x, y, time) of tasks just reached
        self.auction = None          # open auction
        self.auction_seq = 0
        self.task_seq = 0
        self.request_reason = None   # pending auction trigger
        self.last_auction_end = None

        self.announce_pub = self.create_publisher(AuctionAnnouncement, team.ANNOUNCE, team.ANNOUNCE_QOS)
        self.award_pub = self.create_publisher(Award, team.AWARD, team.AWARD_QOS)
        self.viz_pub = self.create_publisher(MarkerArray, '/team/selection_markers', 1)
        self.link_viz_pub = self.create_publisher(MarkerArray, '/team/link_markers', 1)
        self.hb_pub = self.create_publisher(Heartbeat, team.HEARTBEAT, team.HEARTBEAT_QOS)
        self.create_subscription(TaskCandidates, team.CANDIDATES, self.on_candidates, team.CANDIDATES_QOS)
        self.create_subscription(Bid, team.BID, self.on_bid, team.BID_QOS)
        self.create_subscription(TaskStatus, team.TASK_STATUS, self.on_status, team.TASK_STATUS_QOS)
        self.create_subscription(Heartbeat, team.HEARTBEAT, self.on_heartbeat, team.HEARTBEAT_QOS)
        self.create_subscription(LinkQuality, team.LINK_QUALITY, self.on_link, team.LINK_QUALITY_QOS)
        self.create_timer(p('link.fit_period_s', 5.0), self.fit_link_model)

        self.create_timer(self.hb_period, self.publish_heartbeat)
        self.create_timer(0.2, self.tick)
        if not self.joint:
            self.create_timer(self.reevaluate_period, lambda: self.request_auction('periodic re-evaluation'))
        self.goal_seen = {}          # robot -> rclpy Time its goal was last among the candidates
        self.get_logger().info(f'Coordinator for {self.robots} on {self.host_robot}, selection {self.method}'
                               + (f' {self.selection}' if self.joint else ''))

    def _declare(self, name, default):
        return self.declare_parameter(name, default).value

    def now(self):
        return self.get_clock().now()

    def age(self, stamp):
        return (self.now() - stamp).nanoseconds * 1e-9

    def request_auction(self, reason):
        if self.request_reason is None:
            self.request_reason = reason

    # --- inputs ------------------------------------------------------------
    def on_heartbeat(self, msg):
        if msg.role != Heartbeat.ROLE_MEMBER or msg.robot not in self.robots:
            return
        if msg.robot not in self.last_seen:
            self.get_logger().info(f'{msg.robot} joined')
            self.request_auction(f'{msg.robot} joined')
        self.last_seen[msg.robot] = self.now()
        self.robot_pos[msg.robot] = (msg.pose.position.x, msg.pose.position.y)

    def on_link(self, msg):
        if msg.robot not in self.robots:
            return
        self.link_latest[msg.robot] = (self.now(), float(msg.rssi_dbm), msg.connected)
        if msg.connected:
            self.link_model.add(msg.position.x, msg.position.y, float(msg.rssi_dbm))

    def fit_link_model(self):
        m = self.link_model
        m.fit()
        if len(m.rssi) and self.link_viz_pub.get_subscription_count():
            self.link_viz_pub.publish(link_markers(self.world_frame, self.now().to_msg(), m.xy, m.rssi, self.q_min))
        if not m.ready:
            return
        self.link_fits += 1
        if self.link_fits % 12 == 1:   # when it becomes ready, then about every minute
            self.get_logger().info(
                f'Link model: {m.count} samples in {len(m.rssi)} cells, AP ({m.ap[0]:.1f}, {m.ap[1]:.1f}), '
                f'P0 {m.p0:.1f} dBm, n {m.n:.2f}')

    def on_candidates(self, msg):
        if msg.robot in self.robots:
            self.candidates[msg.robot] = (self.now(), list(msg.candidates))

    def on_bid(self, msg):
        if self.auction and msg.auction_id == self.auction['id'] and msg.robot in self.auction['robots']:
            self.auction['bids'][msg.robot] = {i.task_id: i for i in msg.items}

    def on_status(self, msg):
        current = self.assignments.get(msg.robot)
        if current is None or current['task'].id != msg.task_id:
            return
        current['status'] = msg.status
        if msg.status == TaskStatus.FAILED:
            self.record_failure(current['task'].position)
        if msg.status == TaskStatus.SUCCEEDED and current['task'].label != WAIT_LABEL:
            pos = current['task'].position
            self.completed.append((pos.x, pos.y, self.now()))
        if msg.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED, TaskStatus.IDLE):
            self.get_logger().info(
                f'{msg.robot}: task {msg.task_id} {self.status_name(msg.status)} {msg.reason}'.rstrip())
            del self.assignments[msg.robot]
            self.request_auction(f'{msg.robot} needs a task')

    @staticmethod
    def status_name(status):
        return {0: 'accepted', 1: 'active', 2: 'succeeded', 3: 'failed', 4: 'canceled', 5: 'idle'}[status]

    # --- failures ------------------------------------------------------------
    def record_failure(self, pos):
        for f in self.failures:
            if distance((f['x'], f['y']), (pos.x, pos.y)) <= self.fail_radius:
                f['count'] += 1
                f['time'] = self.now()
                break
        else:
            self.failures.append({'x': pos.x, 'y': pos.y, 'count': 1, 'time': self.now()})

    def just_completed(self, x, y):
        # The frontier of a task that was just reached stays in the robots' candidate lists
        # until their map and frontier search catch up; do not hand it out again meanwhile.
        self.completed = [c for c in self.completed if self.age(c[2]) < self.completed_hold]
        return any(distance((cx, cy), (x, y)) <= self.merge_radius for cx, cy, _ in self.completed)

    def failure_at(self, x, y):
        self.failures = [f for f in self.failures if self.age(f['time']) < self.blacklist_duration]
        return max((f['count'] for f in self.failures
                    if distance((f['x'], f['y']), (x, y)) <= self.fail_radius), default=0)

    # --- main loop -------------------------------------------------------------
    def alive_robots(self):
        return [r for r in self.robots if r in self.last_seen and self.age(self.last_seen[r]) < self.hb_timeout]

    def tick(self):
        for robot in list(self.last_seen):
            if self.age(self.last_seen[robot]) >= self.hb_timeout:
                self.get_logger().warn(f'{robot} lost (no heartbeat for {self.hb_timeout:.1f}s)')
                del self.last_seen[robot]
                self.assignments.pop(robot, None)
                self.request_auction(f'{robot} lost')

        alive = self.alive_robots()
        if any(r not in self.assignments for r in alive):
            self.request_auction('idle robot')
        if self.joint:
            self.check_goals_still_frontiers()
            if self.selection.uses_comm_constraint:
                self.check_links()

        if self.auction is not None:
            everyone_bid = all(r in self.auction['bids'] for r in self.auction['robots'])
            if everyone_bid or self.now() >= self.auction['deadline']:
                self.resolve_auction()
            return

        # Do not hammer the robots when there is nothing to hand out.
        if self.request_reason and alive and (
                self.last_auction_end is None or self.age(self.last_auction_end) >= self.bid_window):
            reason, self.request_reason = self.request_reason, None
            self.start_auction(alive, reason)

    def check_goals_still_frontiers(self):
        """Re-select when a busy robot's goal is no longer a frontier for any robot."""
        fresh = [c for stamp, cands in self.candidates.values() if self.age(stamp) <= self.max_age for c in cands]
        for robot, a in list(self.assignments.items()):
            if a['status'] not in BUSY:
                continue
            pos = (a['task'].position.x, a['task'].position.y)
            if any(distance((c.position.x, c.position.y), pos) <= self.merge_radius for c in fresh):
                self.goal_seen[robot] = self.now()
            elif self.age(self.goal_seen.setdefault(robot, self.now())) > self.vanish_timeout:
                self.goal_seen[robot] = self.now()
                self.request_auction(f'{robot} goal no longer a frontier')

    def check_links(self):
        """Re-select when a driving robot's measured link stays below q_min."""
        for robot, a in list(self.assignments.items()):
            if a['status'] not in BUSY or a['task'].label == WAIT_LABEL or robot not in self.link_latest:
                self.link_bad_since.pop(robot, None)
                continue
            _, rssi, connected = self.link_latest[robot]
            if connected and rssi >= self.selection.q_min:
                self.link_bad_since.pop(robot, None)
            elif self.age(self.link_bad_since.setdefault(robot, self.now())) > self.link_drop_s:
                self.link_bad_since[robot] = self.now()
                self.request_auction(f'{robot} link below q_min')

    def link_quality(self, robot, x, y):
        """Predicted RSSI [dBm] at (x, y) in world (Q_ij; the same for every robot with one fixed AP).

        None until the link model has min_samples measurements."""
        return self.link_model.predict(x, y)

    def wait_goal(self, robot):
        """Where a robot without a feasible goal should wait, or None to stay where it is.

        Stays if its own last measurement is good enough, else goes to the nearest place
        measured at least wait_margin above q_min."""
        latest = self.link_latest.get(robot)
        q_min = self.selection.q_min
        if latest is not None and latest[2] and latest[1] >= q_min + self.wait_margin:
            return None
        pos = self.robot_pos.get(robot)
        m = self.link_model
        if pos is None or len(m.rssi) == 0:
            return None
        good = [(x, y) for (x, y), q in zip(m.xy, m.rssi) if q >= q_min + self.wait_margin]
        return min(good, key=lambda p: distance(p, pos)) if good else None

    def start_auction(self, robots, reason):
        pool = []
        for robot, (stamp, cands) in self.candidates.items():
            if robot not in robots or self.age(stamp) > self.max_age:
                continue
            for c in cands:
                if self.failure_at(c.position.x, c.position.y) >= self.max_failures:
                    continue  # blacklisted area
                if self.just_completed(c.position.x, c.position.y):
                    continue
                extra = {'unknown_cells': float(c.unknown_cells),
                         'semantic': float(c.semantic_relevance) if c.semantic_observed else None,
                         # unobserved: rover_vlm sends the robot's mean S over its views (0: none yet)
                         'semantic_prior': float(c.semantic_relevance)
                         if not c.semantic_observed and c.semantic_relevance > 0.0 else None}
                pool.append((robot, c.position.x, c.position.y, c.gain, c.region_id, c.label, extra))
        tasks, self.task_seq = merge_candidates(pool, self.merge_radius, 't', self.task_seq)
        if not tasks:
            self.last_auction_end = self.now()
            return

        self.auction_seq += 1
        deadline = self.now() + Duration(seconds=self.bid_window)
        self.auction = {'id': self.auction_seq, 'robots': list(robots), 'tasks': tasks,
                        'bids': {}, 'deadline': deadline, 'reason': reason}
        msg = AuctionAnnouncement()
        msg.header.stamp = self.now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.auction_id = self.auction_seq
        msg.robots = list(robots)
        msg.deadline = deadline.to_msg()
        for t in tasks:
            c = TaskCandidate(id=t.id, source_robot=','.join(t.sources), gain=float(t.gain),
                              region_id=t.region_id, label=t.label)
            c.position.x, c.position.y = t.x, t.y
            msg.tasks.append(c)
        self.announce_pub.publish(msg)
        log = self.get_logger().debug if reason == 'idle robot' else self.get_logger().info
        log(f'Auction {self.auction_seq} ({reason}): {len(tasks)} tasks, robots {list(robots)}')

    def resolve_auction(self):
        auction, self.auction = self.auction, None
        self.last_auction_end = self.now()
        if self.joint:
            self.resolve_joint(auction)
            return
        tasks = {t.id: t for t in auction['tasks']}

        utilities = {}
        for robot, items in auction['bids'].items():
            for task_id, item in items.items():
                if task_id not in tasks:
                    continue
                u = item.utility
                fails = self.failure_at(tasks[task_id].x, tasks[task_id].y)
                if fails:
                    u = _reduce(u, min(0.9, self.fail_penalty * fails))
                utilities[(robot, task_id)] = u

        # Busy robots: find their current task in this round (same place) and its utility now.
        current, current_match = {}, {}
        for robot, a in self.assignments.items():
            if a['status'] not in BUSY or robot not in auction['bids']:
                continue
            pos = (a['task'].position.x, a['task'].position.y)
            match = min(tasks.values(), key=lambda t: distance((t.x, t.y), pos))
            if distance((match.x, match.y), pos) <= self.merge_radius and (robot, match.id) in utilities:
                current[robot] = (match.id, utilities[(robot, match.id)])
                current_match[robot] = match.id

        awards = ssi_assign(auction['robots'], auction['tasks'], utilities, self.near_radius,
                            self.near_penalty, current, self.switch_threshold)

        for robot in auction['robots']:
            if robot not in auction['bids']:
                continue  # no bid (lost message): leave the robot as it is
            if robot not in awards:
                if self.assignments.get(robot, {}).get('status') not in BUSY:
                    self.assignments.pop(robot, None)
                continue
            task_id, utility = awards[robot]
            if current_match.get(robot) == task_id:
                self.assignments[robot]['utility'] = utility   # keep going, same task id
                continue
            t = tasks[task_id]
            c = TaskCandidate(id=f'{auction["id"]}/{t.id}', source_robot=','.join(t.sources),
                              gain=float(t.gain), region_id=t.region_id, label=t.label)
            c.position.x, c.position.y = t.x, t.y
            self.assignments[robot] = {'task': c, 'utility': utility,
                                       'status': TaskStatus.ACCEPTED, 'auction_id': auction['id']}
            self.get_logger().info(
                f'Auction {auction["id"]}: {robot} -> {c.id} ({t.x:.2f}, {t.y:.2f}) utility {utility:.2f}')
        self.publish_award(auction['id'])

    def resolve_joint(self, auction):
        """Joint goal selection (goal_selection.py) over the robots that bid."""
        tasks = {t.id: t for t in auction['tasks']}
        robots = [r for r in auction['robots'] if r in auction['bids']]
        frontiers = [merge_reports(t.id, t.reports) for t in auction['tasks']]
        by_id = {f.id: f for f in frontiers}
        options = {}
        for robot in robots:
            for task_id, item in auction['bids'][robot].items():
                t = tasks.get(task_id)
                if t is None:
                    continue
                # Only a real Nav2 plan counts; straight-line estimates are "no path".
                options[(robot, task_id)] = Option(
                    path_length=float(item.path_cost) if item.path_planned else None,
                    link_quality=self.link_quality(robot, t.x, t.y))
        # Busy robots whose goal is still offered: keep it unless another is better by keep_bonus.
        current = {}
        for robot in robots:
            cur = self.assignments.get(robot)
            if cur is None or cur['status'] not in BUSY or cur['task'].label == WAIT_LABEL:
                continue
            pos = (cur['task'].position.x, cur['task'].position.y)
            near = [t for t in tasks.values() if distance((t.x, t.y), pos) <= self.merge_radius]
            if near:
                current[robot] = min(near, key=lambda t: distance((t.x, t.y), pos)).id
        chosen, scores = select_goals(self.selection, robots, frontiers, options, current, self.keep_bonus)

        excluded = {}
        for s in scores.values():
            if not s.feasible:
                excluded[s.reason] = excluded.get(s.reason, 0) + 1
        # A robot left waiting re-triggers a selection every bid window: keep those quiet.
        quiet = auction['reason'] == 'idle robot' and not chosen
        (self.get_logger().debug if quiet else self.get_logger().info)(
            f'Selection {auction["id"]} ({self.method}, {auction["reason"]}): {len(frontiers)} frontiers, '
            f'goals {({r: g for r, (g, _) in chosen.items()})}, excluded {excluded or "none"}')

        for robot in robots:
            if robot not in chosen:
                self.assign_wait(robot, auction['id'])
                continue
            task_id, utility = chosen[robot]
            t, s = tasks[task_id], scores[(robot, task_id)]
            cur = self.assignments.get(robot)
            if cur is not None and cur['status'] in BUSY and distance(
                    (cur['task'].position.x, cur['task'].position.y), (t.x, t.y)) <= self.merge_radius:
                cur['utility'] = utility   # same goal: keep driving, keep its id
                continue
            f = by_id[task_id]
            c = TaskCandidate(id=f'{auction["id"]}/{t.id}', source_robot=','.join(t.sources),
                              gain=float(t.gain), region_id=t.region_id, label=t.label,
                              unknown_cells=float(f.unknown_cells))
            c.position.x, c.position.y = t.x, t.y
            if f.semantic is not None:
                c.semantic_relevance, c.semantic_observed = float(f.semantic), True
            self.assignments[robot] = {'task': c, 'utility': utility,
                                       'status': TaskStatus.ACCEPTED, 'auction_id': auction['id']}
            self.goal_seen[robot] = self.now()
            self.get_logger().info(
                f'Selection {auction["id"]}: {robot} -> {c.id} ({t.x:.2f}, {t.y:.2f}) '
                f'U={utility:.3f} S={s.S:.2f} G={s.G:.2f} D={s.D:.2f} Q={s.Q}')
        self.publish_award(auction['id'])
        if self.viz_pub.get_subscription_count():
            waits = {r: (a['task'].position.x, a['task'].position.y) for r, a in self.assignments.items()
                     if a['task'].label == WAIT_LABEL}
            self.viz_pub.publish(selection_markers(self.world_frame, self.now().to_msg(), robots, tasks,
                                                   scores, chosen, self.robot_pos, waits))

    def assign_wait(self, robot, auction_id):
        """No feasible goal: wait where the link is good (comm methods) or stop where it is."""
        target = self.wait_goal(robot) if self.selection.uses_comm_constraint else None
        cur = self.assignments.get(robot)
        if target is None:
            if cur is not None:
                self.get_logger().info(f'Selection {auction_id}: {robot} has no feasible goal, stopping')
            self.assignments.pop(robot, None)
            return
        if cur is not None and cur['task'].label == WAIT_LABEL and cur['status'] in BUSY and distance(
                (cur['task'].position.x, cur['task'].position.y), target) <= self.merge_radius:
            return   # already on its way there
        c = TaskCandidate(id=f'{auction_id}/wait', source_robot='', label=WAIT_LABEL)
        c.position.x, c.position.y = float(target[0]), float(target[1])
        self.assignments[robot] = {'task': c, 'utility': 0.0, 'status': TaskStatus.ACCEPTED,
                                   'auction_id': auction_id}
        self.get_logger().info(f'Selection {auction_id}: {robot} has no feasible goal, '
                               f'waiting at ({target[0]:.2f}, {target[1]:.2f}) where the link was good')

    def publish_award(self, auction_id):
        msg = Award()
        msg.header.stamp = self.now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.auction_id = auction_id
        for robot in self.robots:
            a = Assignment(robot=robot)
            if robot in self.assignments:
                a.task = self.assignments[robot]['task']
                a.utility = float(self.assignments[robot]['utility'])
            msg.assignments.append(a)
        self.award_pub.publish(msg)

    def publish_heartbeat(self):
        msg = Heartbeat()
        msg.header.stamp = self.now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.robot = self.host_robot
        msg.role = Heartbeat.ROLE_COORDINATOR
        msg.mode = Heartbeat.MODE_COORDINATED
        self.hb_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeamCoordinator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
