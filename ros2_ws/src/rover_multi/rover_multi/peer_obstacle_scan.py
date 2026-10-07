"""The other robots as obstacles in this robot's costmaps (virtual LaserScan).

Each robot's costmap only had its own sensors, and the 2D lidar scans ~0.46 m above the floor,
over the other rover's body: the two robots planned straight through each other (2026-10-02).
Their poses are on the shared /tf (world -> <robot>/map -> odom -> base_link), so:

    peer base_link in this robot's base_link (latest TF, at most stale_s old)
        -> peer_scan (sensor_msgs/LaserScan in base_link, 360 deg): beams that hit a disc of
           peer_radius around each peer get the distance to that disc; all other beams are NaN

The costmap source is marking only (no clearing): NaN beams clear nothing, and the old position of
a moving peer is cleared by the lidar's own raytracing like any other vacated cell.
"""
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def disc_ranges(n_beams, peers, radius, max_range):
    """Ranges [n_beams] over 360 deg (beam k at angle -pi + k*2pi/n): distance to the nearest disc
    of `radius` around each (x, y) in `peers` (robot frame), NaN where a beam hits none."""
    ang = -math.pi + np.arange(n_beams) * (2 * math.pi / n_beams)
    out = np.full(n_beams, np.nan)
    for x, y in peers:
        d = math.hypot(x, y)
        if d <= radius or d - radius > max_range:
            continue                               # overlapping (own footprint) or out of range
        c = math.atan2(y, x)
        half = math.asin(radius / d)
        diff = np.abs((ang - c + math.pi) % (2 * math.pi) - math.pi)
        hit = diff <= half
        # distance along the beam to the first intersection with the circle
        s = d * np.cos(diff[hit]) - np.sqrt(np.maximum(radius ** 2 - (d * np.sin(diff[hit])) ** 2, 0.0))
        cur = out[hit]
        out[hit] = np.where(np.isnan(cur), s, np.minimum(cur, s))
    return out


class PeerObstacleScan(Node):
    def __init__(self):
        super().__init__('peer_obstacle_scan')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.robot = p('robot_name', self.get_namespace().strip('/'))
        robots = p('robots', ['robot1', 'robot2'])
        self.peers = [r for r in robots if r != self.robot]
        self.base_frame = p('base_frame', f'{self.robot}/base_link')
        self.radius = p('peer_radius', 0.42)       # half diagonal of the 0.6 x 0.6 m footprint
        self.max_range = p('max_range', 8.0)
        self.stale = p('stale_s', 2.0)             # older peer pose (or clock offset): publish nothing for it
        self.n = int(p('beams', 360))
        self.tf_buffer = Buffer()
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', namespace=self.get_namespace())
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        self.pub = self.create_publisher(LaserScan, p('scan_topic', 'peer_scan'), 5)
        self.create_timer(1.0 / p('rate_hz', 5.0), self.tick)
        self.get_logger().info(f'{self.peers} as obstacles (radius {self.radius} m) in {self.base_frame}')

    def tick(self):
        now = self.get_clock().now()
        pts = []
        for peer in self.peers:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, f'{peer}/base_link', Time())
            except TransformException:
                continue
            age = (now - Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
            if tf.header.stamp.sec and age > self.stale:
                continue
            pts.append((tf.transform.translation.x, tf.transform.translation.y))
        scan = LaserScan()
        scan.header.stamp = now.to_msg()
        scan.header.frame_id = self.base_frame
        scan.angle_min = -math.pi
        scan.angle_increment = 2 * math.pi / self.n
        scan.angle_max = scan.angle_min + scan.angle_increment * (self.n - 1)
        scan.range_min, scan.range_max = 0.05, self.max_range
        scan.ranges = [float(r) for r in disc_ranges(self.n, pts, self.radius, self.max_range)]
        self.pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = PeerObstacleScan()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.tf_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
