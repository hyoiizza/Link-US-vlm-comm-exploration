"""Lower layer of the team, one per robot (/<robot>/team_agent).

* forwards the explorer's ranked frontiers to the coordinator (world frame),
  with their gain re-weighted by the semantic map around them (semantic_gain)
* bids on announced tasks with the path length from this robot's Nav2 planner
* drives to the awarded task with this robot's Nav2 and reports its status
* standalone mode when the coordinator is lost: picks its own frontiers and
  announces them on /team/claim so the other robot stays away

The explorer must run with external_assignment_enabled: true (it only ranks
frontiers; this node owns the Nav2 goals).
"""
import math
import threading

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from frontier_exploration_ros2.msg import FrontierCandidates
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener

from rover_msgs.msg import (
    AuctionAnnouncement, Award, Bid, BidItem, Claim, FrontierRelevance, Heartbeat, SemanticGrid, TaskCandidate,
    TaskCandidates, TaskStatus)
from rover_multi import team_interface as team
from rover_multi.grid_paths import GridPaths
from rover_multi.semantic_gain import gain_factor, SemanticLayer, SemanticWeights

STATUS_NAMES = {0: 'accepted', 1: 'active', 2: 'succeeded', 3: 'failed', 4: 'canceled', 5: 'idle'}


def path_length(path):
    pts = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def initial_turn(pts, yaw, lookahead):
    """|Heading change| [rad] from the robot's yaw to the path [(x, y)] direction `lookahead` m along it."""
    x0, y0 = pts[0]
    target = pts[-1]
    for x, y in pts[1:]:
        if math.hypot(x - x0, y - y0) >= lookahead:
            target = (x, y)
            break
    if math.hypot(target[0] - x0, target[1] - y0) < 1e-3:
        return 0.0
    d = math.atan2(target[1] - y0, target[0] - x0) - yaw
    return abs(math.atan2(math.sin(d), math.cos(d)))


def wait_future(future, timeout):
    """Block the calling (worker) thread until an rclpy future is done."""
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout)


class TeamAgent(Node):
    def __init__(self):
        super().__init__('team_agent')
        p = self._declare
        self.robot = p('robot_name', self.get_namespace().strip('/'))
        self.world_frame = p('world_frame', 'world')
        self.map_frame = p('map_frame', f'{self.robot}/map')
        self.base_frame = p('base_frame', f'{self.robot}/base_link')
        self.hb_period = p('heartbeat.period_s', 1.0)
        self.hb_timeout = p('heartbeat.timeout_s', 4.0)
        self.top_k = p('candidates.top_k', 10)
        self.cand_period = p('candidates.publish_period_s', 2.0)
        self.gain_w = p('bid.gain_weight', 1.0)
        self.dist_w = p('bid.distance_weight', 0.5)
        self.max_planned = p('bid.max_planned', 5)
        self.plan_timeout = p('bid.planner_timeout_s', 1.0)
        self.unplanned_factor = p('bid.unplanned_cost_factor', 1.5)
        self.max_dist = p('bid.max_task_distance_m', 30.0)
        self.turn_weight = p('bid.turn_cost_m_per_rad', 0.0)
        self.turn_lookahead = p('bid.turn_lookahead_m', 0.8)
        # costmap: one Dijkstra over the global costmap for all tasks (grid_paths.py);
        # planner: one Nav2 ComputePathToPose per task (competes with the robot's own navigation)
        self.path_source = p('bid.path_source', 'costmap')
        self.costmap_max_age = p('bid.costmap_max_age_s', 5.0)
        self.grid_downsample = p('bid.grid_downsample', 2)
        self.grid_blocked_cost = p('bid.grid_blocked_cost', 99)
        self.grid_goal_radius = p('bid.grid_goal_radius_m', 0.3)
        self.costmap = None           # (rclpy Time, OccupancyGrid) latest global costmap
        self.claim_radius = p('standalone.claim_radius_m', 3.0)
        self.fail_radius = p('standalone.failure_radius_m', 1.0)
        self.max_failures = p('standalone.max_failures', 2)
        self.blacklist_duration = p('standalone.blacklist_duration_s', 60.0)
        self.sem_weights = SemanticWeights(
            floor=p('semantic.floor_weight', 0.5),
            wall=p('semantic.wall_weight', 0.5),
            object=p('semantic.object_weight', 0.5),
            radius=p('semantic.radius_m', 1.0),
            min_known_cells=p('semantic.min_known_cells', 20),
            min_confidence=p('semantic.min_confidence', 50),
            min_factor=p('semantic.min_factor', 0.2),
            max_factor=p('semantic.max_factor', 2.0))
        self.unknown_radius = p('gain.unknown_radius_m', 1.5)
        self.relevance_match = p('vlm.match_radius_m', 0.3)
        self.relevance_max_age = p('vlm.max_age_s', 5.0)

        self.cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.candidates = []          # [TaskCandidate] in world, best first
        self.coordinator_seen = None  # rclpy Time of the last coordinator heartbeat
        self.start_time = self.get_clock().now()
        self.standalone = False
        self.task = None              # TaskCandidate being driven to (world)
        self.task_auction = 0
        self.goal_handle = None
        self.goal_seq = 0
        self.finished = set()         # task ids already reached/failed (ignore latched re-awards)
        self.claims = {}              # other robot -> Claim
        self.peer_seen = {}           # other robot -> rclpy Time of its last heartbeat
        self.failures = []            # standalone mode: [x, y, count, rclpy Time], by position
        self.cand_seq = 0
        self.semantic = None          # (SemanticLayer, frame_id) of the latest semantic grid
        self.grid = None              # (unknown mask [H, W] bool, resolution, origin x, origin y, frame)
        self.relevance = None         # (rclpy Time, frame, [(x, y, S, observed)]) from rover_vlm

        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb)
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose', callback_group=self.cb)

        self.cand_pub = self.create_publisher(TaskCandidates, team.CANDIDATES, team.CANDIDATES_QOS)
        self.bid_pub = self.create_publisher(Bid, team.BID, team.BID_QOS)
        self.status_pub = self.create_publisher(TaskStatus, team.TASK_STATUS, team.TASK_STATUS_QOS)
        self.hb_pub = self.create_publisher(Heartbeat, team.HEARTBEAT, team.HEARTBEAT_QOS)
        self.claim_pub = self.create_publisher(Claim, team.CLAIM, team.CLAIM_QOS)

        self.create_subscription(
            FrontierCandidates, p('frontier_candidates_topic', 'explore/frontier_candidates'),
            self.on_frontiers, 10, callback_group=self.cb)
        # semantic_mapper publishes its grid latched, so a late start still gets the last one.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(SemanticGrid, p('semantic.grid_topic', 'semantic_mapper/grid'),
                                 self.on_semantic_grid, latched, callback_group=self.cb)
        self.create_subscription(OccupancyGrid, p('gain.map_topic', 'map'), self.on_map, latched,
                                 callback_group=self.cb)
        if self.path_source == 'costmap':
            self.create_subscription(OccupancyGrid, p('bid.costmap_topic', 'global_costmap/costmap'),
                                     self.on_costmap, latched, callback_group=self.cb)
        self.create_subscription(FrontierRelevance, p('vlm.relevance_topic', 'explore/frontier_relevance'),
                                 self.on_relevance, 10, callback_group=self.cb)
        self.create_subscription(AuctionAnnouncement, team.ANNOUNCE, self.on_announce,
                                 team.ANNOUNCE_QOS, callback_group=self.cb)
        self.create_subscription(Award, team.AWARD, self.on_award, team.AWARD_QOS, callback_group=self.cb)
        self.create_subscription(Heartbeat, team.HEARTBEAT, self.on_heartbeat,
                                 team.HEARTBEAT_QOS, callback_group=self.cb)
        self.create_subscription(Claim, team.CLAIM, self.on_claim, team.CLAIM_QOS, callback_group=self.cb)

        self.create_timer(self.hb_period, self.tick, callback_group=self.cb)
        self.create_timer(self.cand_period, self.publish_candidates, callback_group=self.cb)
        self.get_logger().info(f'Team agent for {self.robot} ({self.map_frame} -> {self.world_frame})')

    def _declare(self, name, default):
        return self.declare_parameter(name, default).value

    # --- frames ------------------------------------------------------------------
    def transform_pose(self, pose, source, target):
        tf = self.tf_buffer.lookup_transform(target, source, Time(), Duration(seconds=0.2))
        return do_transform_pose(pose, tf)

    def robot_pose_world(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, Time())
        except TransformException:
            return None
        return tf.transform.translation

    # --- candidates ------------------------------------------------------------
    def on_semantic_grid(self, msg):
        h, w = msg.info.height, msg.info.width
        if h * w == 0 or len(msg.labels) != h * w or len(msg.confidence) != h * w:
            return
        layer = SemanticLayer(
            np.asarray(msg.labels, dtype=np.uint8).reshape(h, w),
            np.asarray(msg.confidence, dtype=np.uint8).reshape(h, w),
            msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y, msg.class_names)
        with self.lock:
            self.semantic = (layer, msg.header.frame_id or self.map_frame)

    def on_map(self, msg):
        h, w = msg.info.height, msg.info.width
        if h * w == 0 or len(msg.data) != h * w:
            return
        unknown = np.asarray(msg.data, dtype=np.int8).reshape(h, w) < 0
        with self.lock:
            self.grid = (unknown, msg.info.resolution, msg.info.origin.position.x,
                         msg.info.origin.position.y, msg.header.frame_id or self.map_frame)

    def unknown_cells(self, goal, frame):
        """Unknown cells of this robot's map within unknown_radius of the goal (raw gain G)."""
        with self.lock:
            grid = self.grid
        if grid is None:
            return 0.0
        unknown, res, x0, y0, grid_frame = grid
        if grid_frame != frame:
            try:
                goal = self.transform_pose(goal, frame, grid_frame)
            except TransformException:
                return 0.0
        r = self.unknown_radius
        gx, gy = goal.position.x, goal.position.y
        h, w = unknown.shape
        i0, i1 = max(int((gy - r - y0) / res), 0), min(int((gy + r - y0) / res) + 1, h)
        j0, j1 = max(int((gx - r - x0) / res), 0), min(int((gx + r - x0) / res) + 1, w)
        if i0 >= i1 or j0 >= j1:
            return 0.0
        ii, jj = np.mgrid[i0:i1, j0:j1]
        disc = np.hypot(x0 + (jj + 0.5) * res - gx, y0 + (ii + 0.5) * res - gy) <= r
        return float(np.count_nonzero(unknown[i0:i1, j0:j1] & disc))

    def semantic_weight(self, goal, frame):
        """(gain factor, label) for a frontier goal from the semantic map around it."""
        with self.lock:
            semantic = self.semantic
        if semantic is None:
            return 1.0, ''
        layer, grid_frame = semantic
        if grid_frame != frame:
            try:
                goal = self.transform_pose(goal, frame, grid_frame)
            except TransformException:
                return 1.0, ''   # no semantic evidence is not a reason to drop the frontier
        w = self.sem_weights
        shares = layer.shares(goal.position.x, goal.position.y, w.radius, w.min_confidence)
        factor = gain_factor(shares, w)
        if shares.known < w.min_known_cells:
            return factor, ''
        return factor, f'floor={shares.floor:.2f} wall={shares.wall:.2f} obj={shares.object:.2f} x{factor:.2f}'

    def on_relevance(self, msg):
        entries = [(p.x, p.y, s, o) for p, s, o in zip(msg.positions, msg.relevance, msg.observed)]
        with self.lock:
            self.relevance = (self.get_clock().now(), msg.header.frame_id or self.map_frame, entries)

    def vlm_relevance(self, goal, frame):
        """(S, observed) the VLM node reported for this frontier goal, (0, False) if none."""
        with self.lock:
            rel = self.relevance
        if rel is None or (self.get_clock().now() - rel[0]).nanoseconds * 1e-9 > self.relevance_max_age:
            return 0.0, False
        stamp, rel_frame, entries = rel
        if rel_frame != frame:
            try:
                goal = self.transform_pose(goal, frame, rel_frame)
            except TransformException:
                return 0.0, False
        best = min(entries, key=lambda e: math.hypot(e[0] - goal.position.x, e[1] - goal.position.y),
                   default=None)
        if best is None or math.hypot(best[0] - goal.position.x, best[1] - goal.position.y) > self.relevance_match:
            return 0.0, False
        return float(best[2]), bool(best[3])

    def on_frontiers(self, msg):
        out = []
        frame = msg.header.frame_id or self.map_frame
        try:
            for i, (goal, gain) in enumerate(zip(msg.goals[:self.top_k], msg.gains)):
                factor, label = self.semantic_weight(goal, frame)
                S, observed = self.vlm_relevance(goal, frame)
                w = self.transform_pose(goal, frame, self.world_frame)
                c = TaskCandidate(id=f'{self.robot}/{self.cand_seq}.{i}', source_robot=self.robot,
                                  gain=float(gain * factor), label=label,
                                  unknown_cells=self.unknown_cells(goal, frame),
                                  semantic_relevance=S, semantic_observed=observed)
                c.position = w.position
                out.append(c)
        except TransformException as e:
            self.get_logger().warn(f'Cannot place frontiers in {self.world_frame}: {e}',
                                   throttle_duration_sec=5.0)
            return
        self.cand_seq += 1
        with self.lock:
            self.candidates = out

    def publish_candidates(self):
        with self.lock:
            cands = list(self.candidates)
        msg = TaskCandidates(robot=self.robot, candidates=cands)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        self.cand_pub.publish(msg)

    # --- bidding ------------------------------------------------------------------
    def on_announce(self, msg):
        if self.robot not in msg.robots or self.standalone:
            return
        threading.Thread(target=self.bid, args=(msg,), daemon=True).start()

    def on_costmap(self, msg):
        with self.lock:
            self.costmap = (self.get_clock().now(), msg)

    def grid_costs(self, tasks):
        """{task id: path cost} from one Dijkstra over the global costmap (unreachable tasks
        absent), or None when there is no fresh costmap / robot pose."""
        with self.lock:
            cm = self.costmap
        if cm is None or (self.get_clock().now() - cm[0]).nanoseconds * 1e-9 > self.costmap_max_age:
            return None
        msg = cm[1]
        frame = msg.header.frame_id or self.map_frame
        try:
            robot = self.tf_buffer.lookup_transform(frame, self.base_frame, Time())
            to_grid = self.tf_buffer.lookup_transform(frame, self.world_frame, Time())
        except TransformException:
            return None
        info = msg.info
        cost = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        rt, q = robot.transform.translation, robot.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        grid = GridPaths(cost, info.resolution, (info.origin.position.x, info.origin.position.y),
                         (rt.x, rt.y), blocked_cost=self.grid_blocked_cost, downsample=self.grid_downsample)
        costs = {}
        for task in tasks:
            pose = PoseStamped().pose
            pose.position = task.position
            pose.orientation.w = 1.0
            g = do_transform_pose(pose, to_grid).position
            found = grid.path_to(g.x, g.y, self.grid_goal_radius)
            if found is None:
                continue
            length, pts = found
            if self.turn_weight > 0.0:
                length += self.turn_weight * initial_turn(pts, yaw, self.turn_lookahead)
            costs[task.id] = length
        return costs

    def robot_yaw_map(self):
        try:
            q = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time()).transform.rotation
        except TransformException:
            return None
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def plan_cost(self, task):
        """Nav2 path length to the task [m] + turn_weight * initial heading change, or None.

        The turn term makes goals behind the robot more expensive, so a re-selection does
        not keep sending it back the way it came."""
        if not self.planner.wait_for_server(timeout_sec=0.2):
            return None
        goal = ComputePathToPose.Goal()
        goal.goal = PoseStamped()
        goal.goal.header.frame_id = self.world_frame
        goal.goal.pose.position = task.position
        goal.goal.pose.orientation.w = 1.0
        try:
            goal.goal.pose = self.transform_pose(goal.goal.pose, self.world_frame, self.map_frame)
        except TransformException:
            return None
        goal.goal.header.frame_id = self.map_frame
        goal.use_start = False
        send = self.planner.send_goal_async(goal)
        if not wait_future(send, self.plan_timeout) or not send.result().accepted:
            return None
        result = send.result().get_result_async()
        if not wait_future(result, self.plan_timeout):
            send.result().cancel_goal_async()
            return None
        res = result.result()
        if res.status != GoalStatus.STATUS_SUCCEEDED or len(res.result.path.poses) < 2:
            return None
        cost = path_length(res.result.path)
        yaw = self.robot_yaw_map() if self.turn_weight > 0.0 else None
        if yaw is not None:
            pts = [(p.pose.position.x, p.pose.position.y) for p in res.result.path.poses]
            cost += self.turn_weight * initial_turn(pts, yaw, self.turn_lookahead)
        return cost

    def bid(self, msg):
        pos = self.robot_pose_world()
        if pos is None:
            return
        straight = {t.id: math.hypot(t.position.x - pos.x, t.position.y - pos.y) for t in msg.tasks}
        reachable = sorted((t for t in msg.tasks if straight[t.id] <= self.max_dist),
                           key=lambda t: straight[t.id])
        out = Bid(auction_id=msg.auction_id, robot=self.robot)
        grid = self.grid_costs(reachable) if self.path_source == 'costmap' else None
        if grid is not None:
            for task in reachable:
                planned = task.id in grid
                cost = grid[task.id] if planned else straight[task.id] * self.unplanned_factor
                out.items.append(BidItem(task_id=task.id, path_cost=float(cost), path_planned=planned,
                                         utility=float(self.gain_w * task.gain - self.dist_w * cost)))
            reachable = []        # planner fallback below only without a costmap
        for rank, task in enumerate(reachable):
            if self.get_clock().now() >= Time.from_msg(msg.deadline):
                break
            cost = self.plan_cost(task) if rank < self.max_planned else None
            planned = cost is not None
            if not planned:
                cost = straight[task.id] * self.unplanned_factor
            out.items.append(BidItem(task_id=task.id, path_cost=float(cost), path_planned=planned,
                                     utility=float(self.gain_w * task.gain - self.dist_w * cost)))
        out.header.stamp = self.get_clock().now().to_msg()
        self.bid_pub.publish(out)

    # --- executing tasks -------------------------------------------------------------
    def on_award(self, msg):
        if self.standalone:
            return
        mine = next((a for a in msg.assignments if a.robot == self.robot), None)
        if mine is None:
            return
        if not mine.task.id:
            if self.task is not None:
                self.cancel_task('no task in the latest award')
            return
        if mine.task.id in self.finished or (self.task is not None and self.task.id == mine.task.id):
            return
        self.start_task(mine.task, msg.auction_id)

    def start_task(self, task, auction_id):
        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.pose.position = task.position
        pose.pose.orientation.w = 1.0
        try:
            pose.pose = self.transform_pose(pose.pose, self.world_frame, self.map_frame)
        except TransformException as e:
            self.report(task, auction_id, TaskStatus.FAILED, f'no transform: {e}')
            return
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        if not self.nav.wait_for_server(timeout_sec=1.0):
            self.report(task, auction_id, TaskStatus.FAILED, 'navigate_to_pose not available')
            return

        with self.lock:
            self.goal_seq += 1
            seq = self.goal_seq
            previous, self.task, self.task_auction = self.task, task, auction_id
        if previous is not None and previous.id != task.id:
            self.report(previous, auction_id, TaskStatus.CANCELED, f'replaced by {task.id}')
            self.release_claim(previous)
        self.get_logger().info(
            f'Task {task.id}: ({task.position.x:.2f}, {task.position.y:.2f}) in {self.world_frame}')
        self.report(task, auction_id, TaskStatus.ACCEPTED)
        if self.standalone:
            self.publish_claim(task, release=False)

        goal = NavigateToPose.Goal(pose=pose)
        future = self.nav.send_goal_async(goal)
        future.add_done_callback(lambda f: self.on_goal_response(f, seq, task, auction_id))

    def on_goal_response(self, future, seq, task, auction_id):
        handle = future.result()
        if seq != self.goal_seq:
            return
        if not handle.accepted:
            self.finish(seq, task, auction_id, TaskStatus.FAILED, 'goal rejected by Nav2')
            return
        self.goal_handle = handle
        self.report(task, auction_id, TaskStatus.ACTIVE)
        handle.get_result_async().add_done_callback(
            lambda f: self.on_result(f, seq, task, auction_id))

    def on_result(self, future, seq, task, auction_id):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.finish(seq, task, auction_id, TaskStatus.SUCCEEDED)
        elif status == GoalStatus.STATUS_CANCELED:
            self.finish(seq, task, auction_id, TaskStatus.CANCELED, 'canceled')
        else:
            self.finish(seq, task, auction_id, TaskStatus.FAILED, f'Nav2 status {status}')

    def finish(self, seq, task, auction_id, status, reason=''):
        with self.lock:
            if seq != self.goal_seq:
                return  # an older goal that was replaced
            self.task = None
            self.goal_handle = None
            self.finished.add(task.id)
        self.report(task, auction_id, status, reason)
        self.release_claim(task)
        if status == TaskStatus.FAILED:
            self.record_failure(task.position)
        if self.standalone:
            self.pick_standalone_task()

    def record_failure(self, pos):
        # Standalone mode has no coordinator to blacklist places, and candidate ids change
        # with every frontier update, so failures are remembered by position.
        now = self.get_clock().now()
        for f in self.failures:
            if math.hypot(f[0] - pos.x, f[1] - pos.y) <= self.fail_radius:
                f[2] += 1
                f[3] = now
                return
        self.failures.append([pos.x, pos.y, 1, now])

    def blacklisted(self, pos):
        now = self.get_clock().now()
        self.failures = [f for f in self.failures
                         if (now - f[3]).nanoseconds * 1e-9 < self.blacklist_duration]
        return any(f[2] >= self.max_failures and math.hypot(f[0] - pos.x, f[1] - pos.y) <= self.fail_radius
                   for f in self.failures)

    def cancel_task(self, reason):
        with self.lock:
            handle, task = self.goal_handle, self.task
        if task is not None:
            self.get_logger().info(f'Canceling task {task.id}: {reason}')
        if handle is not None:
            handle.cancel_goal_async()

    def report(self, task, auction_id, status, reason=''):
        msg = TaskStatus(robot=self.robot, auction_id=auction_id, task_id=task.id, status=status, reason=reason)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(msg)
        if status not in (TaskStatus.ACCEPTED, TaskStatus.ACTIVE):
            self.get_logger().info(f'Task {task.id} {STATUS_NAMES[status]} {reason}'.rstrip())

    # --- heartbeat / standalone mode ----------------------------------------------------
    def on_heartbeat(self, msg):
        if msg.role == Heartbeat.ROLE_COORDINATOR:
            self.coordinator_seen = self.get_clock().now()
        elif msg.robot != self.robot:
            self.peer_seen[msg.robot] = self.get_clock().now()

    def on_claim(self, msg):
        if msg.robot == self.robot:
            return
        if msg.release:
            self.claims.pop(msg.robot, None)
        else:
            self.claims[msg.robot] = msg

    def publish_claim(self, task, release):
        msg = Claim(robot=self.robot, task=task, release=release)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        self.claim_pub.publish(msg)

    def release_claim(self, task):
        if self.standalone:
            self.publish_claim(task, release=True)

    def tick(self):
        now = self.get_clock().now()
        reference = self.coordinator_seen or self.start_time
        lost = (now - reference).nanoseconds * 1e-9 >= self.hb_timeout
        if lost and not self.standalone:
            self.get_logger().warn('Coordinator lost: switching to standalone exploration')
            self.standalone = True
            if self.task is not None:
                self.publish_claim(self.task, release=False)
            else:
                self.pick_standalone_task()
        elif not lost and self.standalone:
            self.get_logger().info('Coordinator back: following its awards again')
            self.standalone = False
            if self.task is not None:
                self.publish_claim(self.task, release=True)
        elif self.standalone and self.task is None:
            self.pick_standalone_task()
        self.publish_heartbeat()

    def pick_standalone_task(self):
        pos = self.robot_pose_world()
        with self.lock:
            cands = list(self.candidates)
        if pos is None or not cands:
            return
        # Claims of robots that went silent are stale (latched on /team/claim): ignore them.
        now = self.get_clock().now()
        claimed = [c.task.position for r, c in self.claims.items()
                   if r in self.peer_seen and (now - self.peer_seen[r]).nanoseconds * 1e-9 < self.hb_timeout]

        def free(c):
            return all(math.hypot(c.position.x - q.x, c.position.y - q.y) > self.claim_radius for q in claimed)

        options = [c for c in cands if free(c) and not self.blacklisted(c.position)]
        if not options:
            return
        best = max(options, key=lambda c: self.gain_w * c.gain - self.dist_w * self.unplanned_factor *
                   math.hypot(c.position.x - pos.x, c.position.y - pos.y))
        self.start_task(best, 0)

    def publish_heartbeat(self):
        msg = Heartbeat(robot=self.robot, role=Heartbeat.ROLE_MEMBER,
                        mode=Heartbeat.MODE_STANDALONE if self.standalone else Heartbeat.MODE_COORDINATED)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        pos = self.robot_pose_world()
        if pos is not None:
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos.x, pos.y, pos.z
            msg.pose.orientation.w = 1.0
        msg.current_task_id = self.task.id if self.task is not None else ''
        self.hb_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeamAgent()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
