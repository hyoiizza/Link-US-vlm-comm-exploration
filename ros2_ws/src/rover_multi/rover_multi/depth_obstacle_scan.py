"""Low obstacles from the depth camera as a LaserScan for the Nav2 costmaps.

The 2D lidar scans ~0.46 m above the floor (base_link is ~0.43 m up, the lidar 0.033 m
above it), so boxes, steps and chair bases below that plane never reach the costmap and
the planner drives into them. The Gemini 335 looks straight ahead from ~0.63 m and sees
the floor from ~1.2 m in front of the robot outward.

    camera/depth/image_raw + camera/depth/camera_info
        -> points (every `stride`-th pixel) in base_link
        -> keep  floor_z + min_height  <  z  <  floor_z + max_height,  0 < x,  range < max_range
        -> depth_scan (sensor_msgs/LaserScan in base_link, one beam per angle_step over the
           camera's horizontal field of view): nearest kept point per beam; beams that saw
           floor but nothing in the band report +inf (free, clears the costmap up to max_range)

Costmap side (params/*.yaml, observation source depth_scan): raytrace_min_range ~1.3 m so
obstacles that drop into the camera's blind zone in front of the robot are never cleared.
Images arrive raw and are decoded at most `rate_hz` times a second.
"""
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def depth_to_scan(depth_m, K, R, t, floor_z, min_h, max_h, max_range, angle_min, angle_max, step, stride=4):
    """Ranges [m] per beam (nan = beam saw nothing, inf = saw floor only) from one depth image.

    depth_m: [H, W] metres (0 = invalid); K: (fx, fy, cx, cy); R, t: camera optical -> base_link."""
    fx, fy, cx, cy = K
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    ok = (d > 0.15) & (d < max_range + 1.0)
    z = d[ok]
    pc = np.stack([(u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z])
    pb = R @ pc + t[:, None]                       # base_link
    x, y, h = pb
    rng = np.hypot(x, y)
    ang = np.arctan2(y, x)
    n = int(round((angle_max - angle_min) / step)) + 1
    beam = np.round((ang - angle_min) / step).astype(int)
    inside = (beam >= 0) & (beam < n) & (x > 0) & (rng <= max_range)
    out = np.full(n, np.nan)
    floor = inside & (np.abs(h - floor_z) < min_h)
    out[np.unique(beam[floor])] = np.inf
    obst = inside & (h > floor_z + min_h) & (h < floor_z + max_h)
    if obst.any():
        b, r = beam[obst], rng[obst]
        order = np.lexsort((r, b))
        b, r = b[order], r[order]
        first = np.r_[True, b[1:] != b[:-1]]
        out[b[first]] = r[first]                  # nearest obstacle per beam wins over floor
    return out


class DepthObstacleScan(Node):
    def __init__(self):
        super().__init__('depth_obstacle_scan')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.base_frame = p('base_frame', 'base_link')
        self.floor_z = p('floor_z', -0.43)          # floor height in base_link [m] (measured 2026-10-01)
        self.min_h = p('min_height', 0.05)          # above the floor: below is floor / noise
        self.max_h = p('max_height', 0.80)          # above the floor: higher does not hit the robot
        self.max_range = p('max_range', 3.0)
        self.step = math.radians(p('angle_step_deg', 1.0))
        self.stride = p('stride', 4)
        self.period = 1.0 / p('rate_hz', 5.0)
        self.K = None
        self.last = 0.0
        self.tf_buffer = Buffer()
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', namespace=self.get_namespace())
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        self.extrinsic = None
        self.pub = self.create_publisher(LaserScan, p('scan_topic', 'depth_scan'), 5)
        self.create_subscription(CameraInfo, p('camera_info_topic', 'camera/depth/camera_info'),
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, p('depth_topic', 'camera/depth/image_raw'), self.on_depth,
                                 qos_profile_sensor_data, raw=True)
        self.get_logger().info(f'Low obstacles from depth: {self.min_h:.2f}-{self.max_h:.2f} m above the floor '
                               f'(floor z {self.floor_z:+.2f} in {self.base_frame}), up to {self.max_range:.1f} m')

    def on_info(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5], msg.width)

    def on_depth(self, raw):
        now = self.get_clock().now()
        if self.K is None or now.nanoseconds * 1e-9 - self.last < self.period:
            return
        self.last = now.nanoseconds * 1e-9
        msg = deserialize_message(raw, Image)
        if self.extrinsic is None or self.extrinsic[0] != msg.header.frame_id:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
            except TransformException as e:
                self.get_logger().warn(f'No {msg.header.frame_id} -> {self.base_frame}: {e}', throttle_duration_sec=5.0)
                return
            tr = tf.transform.translation
            self.extrinsic = (msg.header.frame_id, quaternion_matrix(tf.transform.rotation),
                              np.array([tr.x, tr.y, tr.z]))
            # horizontal field of view of the camera, in base_link
            fx, _, cx, _, w = self.K
            half = max(math.atan2(cx, fx), math.atan2(w - cx, fx))
            self.angle_min, self.angle_max = -half, half
        _, R, t = self.extrinsic
        if msg.encoding in ('16UC1', 'mono16'):
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, -1)[:, :msg.width] * 0.001
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, -1)[:, :msg.width].astype(np.float64)
        else:
            self.get_logger().warn(f'Unsupported depth encoding {msg.encoding}', throttle_duration_sec=10.0)
            return
        if depth.shape[1] != self.K[4]:
            return
        ranges = depth_to_scan(depth, self.K[:4], R, t, self.floor_z, self.min_h, self.max_h, self.max_range,
                               self.angle_min, self.angle_max, self.step, self.stride)
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = self.base_frame
        scan.angle_min, scan.angle_max, scan.angle_increment = self.angle_min, self.angle_min + self.step * (len(ranges) - 1), self.step
        scan.range_min, scan.range_max = 0.05, self.max_range
        scan.ranges = [float(r) for r in ranges]     # nan: no data for that beam (ignored by the costmap)
        self.pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = DepthObstacleScan()
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
