"""Lift 2D YOLO detections to 3D using the registered depth image and keep a
persistent list of person / fire locations in the map frame (RTAB-Map).

Inputs  : ~/detections (Detection2DArray), depth image aligned to color, color camera_info
Outputs : ~/detections_3d (Detection3DArray, map frame, current frame only)
          ~/objects       (MarkerArray, map frame, all merged objects)
"""
import struct
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from rover_object.image_conv import to_array

COLORS = {'person': (0.1, 0.9, 0.1), 'fire': (1.0, 0.2, 0.0)}


class MappedObject:
    def __init__(self, obj_id, label, pos, score, stamp):
        self.id = obj_id
        self.label = label
        self.pos = np.array(pos, dtype=np.float64)
        self.hits = 1
        self.score = score
        self.last_seen = stamp

    def update(self, pos, score, stamp, alpha):
        self.pos = (1.0 - alpha) * self.pos + alpha * np.asarray(pos)
        self.hits += 1
        self.score = max(self.score, score)
        self.last_seen = stamp


class ObjectMapperNode(Node):
    def __init__(self):
        super().__init__('object_mapper')
        self.declare_parameter('detections_topic', 'object_detector/detections')
        self.declare_parameter('depth_topic', 'camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', 'camera/color/camera_info')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('depth_scale', 0.001)       # 16UC1 depth in mm -> m
        self.declare_parameter('min_depth', 0.2)
        self.declare_parameter('max_depth', 8.0)
        self.declare_parameter('roi_scale', 0.4)           # central fraction of the bbox used for depth
        self.declare_parameter('merge_distance.person', 1.0)
        self.declare_parameter('merge_distance.fire', 0.7)
        self.declare_parameter('min_hits', 3)              # detections before an object is published
        self.declare_parameter('position_alpha', 0.3)      # EMA weight for new observations
        self.declare_parameter('person_timeout', 10.0)     # people move: drop if unseen [s] (0 = keep)
        self.declare_parameter('fire_timeout', 0.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame = p('map_frame')
        self.depth_scale = p('depth_scale')
        self.min_depth, self.max_depth = p('min_depth'), p('max_depth')
        self.roi_scale = p('roi_scale')
        self.merge_dist = {'person': p('merge_distance.person'), 'fire': p('merge_distance.fire')}
        self.min_hits = p('min_hits')
        self.alpha = p('position_alpha')
        self.timeout = {'person': p('person_timeout'), 'fire': p('fire_timeout')}

        self.K = None
        self.cam_frame = None
        self.objects = []
        self.next_id = 0

        self.tf_buffer = Buffer()
        # /tf on its own node + thread: on this node's executor depth/detection callbacks starved it
        # (same fix as rover_vlm, 2026-10-01).
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', namespace=self.get_namespace())
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)

        self.create_subscription(CameraInfo, p('camera_info_topic'), self.info_cb, qos_profile_sensor_data)
        # Depth frames are kept serialized and only the one matching a detection is decoded:
        # decoding every 15 fps frame in Python (message_filters) loaded the Jetson for nothing.
        self.depth_buf = deque(maxlen=20)         # (stamp [s], serialized Image)
        self.slop = 0.05
        self.create_subscription(Image, p('depth_topic'), self.depth_cb, qos_profile_sensor_data, raw=True)
        self.create_subscription(Detection2DArray, p('detections_topic'), self.det_cb, 10)

        self.det3d_pub = self.create_publisher(Detection3DArray, '~/detections_3d', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/objects', 10)
        self.create_timer(1.0, self.publish_markers)

    def depth_cb(self, raw):
        endian = '<' if raw[1] == 1 else '>'      # CDR header: stamp right after the 4-byte encapsulation
        sec, nanosec = struct.unpack_from(endian + 'iI', raw, 4)
        self.depth_buf.append((sec + nanosec * 1e-9, raw))

    def det_cb(self, dets):
        if not dets.detections or not self.depth_buf:
            return
        t = dets.header.stamp.sec + dets.header.stamp.nanosec * 1e-9
        stamp, raw = min(self.depth_buf, key=lambda e: abs(e[0] - t))
        if abs(stamp - t) <= self.slop:
            self.sync_cb(dets, deserialize_message(raw, Image))

    def info_cb(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.cam_frame = msg.header.frame_id

    def _depth_at(self, depth, cx, cy, w, h):
        hw, hh = max(1, int(w * self.roi_scale / 2)), max(1, int(h * self.roi_scale / 2))
        H, W = depth.shape[:2]
        x0, x1 = max(0, int(cx) - hw), min(W, int(cx) + hw + 1)
        y0, y1 = max(0, int(cy) - hh), min(H, int(cy) + hh + 1)
        roi = depth[y0:y1, x0:x1].astype(np.float32)
        if depth.dtype == np.uint16:
            roi *= self.depth_scale
        roi = roi[np.isfinite(roi) & (roi > self.min_depth) & (roi < self.max_depth)]
        if roi.size < 5:
            return None
        # Closer percentile: the target usually occludes the background inside its box.
        return float(np.percentile(roi, 30))

    def sync_cb(self, dets, depth_msg):
        if self.K is None or not dets.detections:
            return
        depth = to_array(depth_msg)
        # Depth is registered to color, so its pixels live in the color optical frame.
        cam_frame = self.cam_frame or depth_msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, cam_frame, Time.from_msg(depth_msg.header.stamp), Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warn(f'TF {cam_frame} -> {self.map_frame} unavailable: {e}',
                                   throttle_duration_sec=5.0)
            return

        fx, fy, cx0, cy0 = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        now = Time.from_msg(depth_msg.header.stamp)
        out = Detection3DArray()
        out.header.stamp = depth_msg.header.stamp
        out.header.frame_id = self.map_frame

        for det in dets.detections:
            if not det.results:
                continue
            label = det.results[0].hypothesis.class_id
            score = det.results[0].hypothesis.score
            u, v = det.bbox.center.position.x, det.bbox.center.position.y
            z = self._depth_at(depth, u, v, det.bbox.size_x, det.bbox.size_y)
            if z is None:
                continue

            pt = PointStamped()
            pt.header.stamp = depth_msg.header.stamp
            pt.header.frame_id = cam_frame
            pt.point.x = (u - cx0) * z / fx
            pt.point.y = (v - cy0) * z / fy
            pt.point.z = z
            pm = do_transform_point(pt, tf).point
            pos = (pm.x, pm.y, pm.z)

            obj = self._associate(label, pos, score, now)

            d3 = Detection3D()
            d3.header = out.header
            d3.id = str(obj.id)
            d3.bbox.center.position = pm
            d3.bbox.size.x = d3.bbox.size.y = det.bbox.size_x * z / fx
            d3.bbox.size.z = det.bbox.size_y * z / fy
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score
            hyp.pose.pose.position = pm
            d3.results.append(hyp)
            out.detections.append(d3)

        self.det3d_pub.publish(out)

    def _associate(self, label, pos, score, stamp):
        best, best_d = None, self.merge_dist.get(label, 1.0)
        for obj in self.objects:
            if obj.label != label:
                continue
            d = float(np.linalg.norm(obj.pos - pos))
            if d < best_d:
                best, best_d = obj, d
        if best is None:
            best = MappedObject(self.next_id, label, pos, score, stamp)
            self.next_id += 1
            self.objects.append(best)
            self.get_logger().info(f'New {label} #{best.id} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})')
        else:
            best.update(pos, score, stamp, self.alpha)
        return best

    def publish_markers(self):
        now = self.get_clock().now()
        kept = []
        for obj in self.objects:
            t = self.timeout.get(obj.label, 0.0)
            if t > 0 and (now - obj.last_seen).nanoseconds * 1e-9 > t:
                continue
            kept.append(obj)
        self.objects = kept

        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for obj in self.objects:
            if obj.hits < self.min_hits:
                continue
            r, g, b = COLORS.get(obj.label, (0.2, 0.4, 1.0))
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now.to_msg()
            m.ns = obj.label
            m.id = obj.id
            m.type = Marker.SPHERE if obj.label == 'fire' else Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = obj.pos
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.4
            m.scale.z = 0.4 if obj.label == 'fire' else 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.8
            arr.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = obj.label + '_label'
            t.id = obj.id
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x, t.pose.position.y = obj.pos[0], obj.pos[1]
            t.pose.position.z = obj.pos[2] + 0.8
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f'{obj.label} #{obj.id} ({obj.score:.2f})'
            arr.markers.append(t)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tf_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
