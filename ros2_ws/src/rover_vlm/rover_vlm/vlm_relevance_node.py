"""Search-task relevance S_j of this robot's frontiers from its own camera keyframes.

    camera/color/image_raw + camera_info (+ camera/depth/image_raw for occlusion)
        -> keyframe every min_translation / min_rotation: n crops -> TensorRT image
           encoder -> S per crop (two-prompt softmax, see relevance.py)
    explore/frontier_candidates
        -> explore/frontier_relevance: S of the crop that sees each goal best

Only S values leave the robot; images stay on the Jetson. The engine and the JSON
(prompts, text embeddings, preprocessing) come from scripts/export_siglip2.py.
"""
import os
import struct
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from frontier_exploration_ros2.msg import FrontierCandidates
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from rover_msgs.msg import FrontierRelevance
from rover_vlm.image_conv import to_array, to_bgr
from rover_vlm.relevance import (
    crop_columns, depth_profile, KeyframeStore, prompt_embedding, relevance, ViewParams)


def header_of(raw):
    """(stamp [s], frame_id) of a serialized message that starts with std_msgs/Header.

    Images arrive as raw CDR bytes: deserializing every 848x480 color + depth frame in Python
    (~30 MB/s at 15 fps) held the GIL so long that the TF thread fell 8-30 s behind and
    almost no keyframe got a camera pose (run 2026-10-01 01:51). Only keyframes are decoded."""
    endian = '<' if raw[1] == 1 else '>'           # CDR encapsulation: 0x0001 little endian
    sec, nanosec, n = struct.unpack_from(endian + 'iII', raw, 4)
    return sec + nanosec * 1e-9, raw[16:16 + n - 1].decode()


def quaternion_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class VlmRelevanceNode(Node):
    def __init__(self, encoder=None, meta=None):
        super().__init__('vlm_relevance')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        models = os.path.join(get_package_share_directory('rover_vlm'), 'models')
        engine = p('engine_path', '') or os.path.join(models, 'siglip2_base_patch16_224_fp16.engine')
        meta_path = p('meta_path', '') or os.path.join(models, 'siglip2_base_patch16_224.json')
        self.map_frame = p('map_frame', 'map')
        self.odom_frame = p('odom_frame', 'odom')
        self.n_crops = p('n_crops', 3)
        similarity = p('similarity', 'scaled')     # scaled: model logit scale, cosine: scale 1
        temperature = p('temperature', 1.0)
        self.min_period = 1.0 / p('max_keyframe_rate_hz', 2.0)
        # percentile: S = rank of the frontier's crop among every crop scored so far (relevance.py)
        self.normalization = p('normalization', 'percentile')
        if self.normalization not in ('percentile', 'none'):
            raise ValueError(f'normalization {self.normalization!r}: expected percentile or none')
        self.min_reference = p('min_reference_crops', 30)
        self.depth_max_dt = p('depth_max_dt_s', 0.15)
        self.view = ViewParams(
            min_range=p('min_range_m', 0.5), max_range=p('max_range_m', 6.0),
            occlusion_margin=p('occlusion_margin_m', 0.5),
            min_translation=p('min_translation_m', 0.5), min_rotation_deg=p('min_rotation_deg', 15.0),
            max_keyframes=p('max_keyframes', 500))
        self.store = KeyframeStore(self.view)

        if encoder is None:
            from rover_vlm.trt_image_encoder import load_meta, TrtImageEncoder
            meta = load_meta(meta_path)
            encoder = TrtImageEncoder(engine, meta)
        self.encoder = encoder
        self.pos = prompt_embedding(meta['text_embeds']['positive'])
        self.neg = prompt_embedding(meta['text_embeds']['negative'])
        self.scale = float(meta['logit_scale']) if similarity == 'scaled' else 1.0
        self.temperature = temperature
        self.get_logger().info(
            f'{meta["model_id"]} ({meta["image_size"]}px), {self.n_crops} crops/keyframe, '
            f'similarity {similarity} (scale {self.scale:.2f}, temperature {temperature}), '
            f'prompts +{meta.get("prompts", {}).get("positive")} -{meta.get("prompts", {}).get("negative")}')

        self.K = None
        self.depth = None             # latest serialized depth Image, decoded only for a keyframe
        self.last_keyframe = 0.0
        self.tf_buffer = Buffer()
        # /tf on its own node and thread. spin_thread=True on this node is not enough: Humble's
        # listener adds the whole node to its executor, so that thread also ran image callbacks
        # and inference and the buffer fell 8-30 s behind (run 2026-10-01 01:51).
        self.tf_node = rclpy.create_node(f'{self.get_name()}_tf', namespace=self.get_namespace())
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        self.pub = self.create_publisher(FrontierRelevance, 'explore/frontier_relevance', 10)
        self.create_subscription(CameraInfo, p('camera_info_topic', 'camera/color/camera_info'),
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, p('image_topic', 'camera/color/image_raw'),
                                 self.on_image, qos_profile_sensor_data, raw=True)
        if p('use_depth_occlusion', True):
            self.create_subscription(Image, p('depth_topic', 'camera/depth/image_raw'),
                                     self.on_depth, qos_profile_sensor_data, raw=True)
        self.create_subscription(FrontierCandidates, p('frontier_candidates_topic', 'explore/frontier_candidates'),
                                 self.on_frontiers, 10)

    def on_info(self, msg):
        self.K = (msg.k[0], msg.k[2], msg.width, msg.height)

    def on_depth(self, msg):
        self.depth = msg

    def on_image(self, raw):
        now = time.monotonic()
        if self.K is None or now - self.last_keyframe < self.min_period:
            return
        stamp, frame_id = header_of(raw)
        stamp_time = Time(nanoseconds=int(round(stamp * 1e9)))
        # Camera pose at the image stamp from odom (EKF, 30 Hz), then the latest map -> odom:
        # rtabmap updates map -> odom only about once a second, so a direct map lookup at the
        # image stamp failed with "extrapolation into the future" and no keyframe was ever taken.
        try:
            odom_cam = self.tf_buffer.lookup_transform(self.odom_frame, frame_id, stamp_time,
                                                       Duration(seconds=0.2))
            map_odom = self.tf_buffer.lookup_transform(self.map_frame, self.odom_frame, Time())
        except TransformException as e:
            self.get_logger().warn(f'TF {frame_id} -> {self.map_frame}: {e}', throttle_duration_sec=5.0)
            return
        Ro, to = quaternion_matrix(odom_cam.transform.rotation), odom_cam.transform.translation
        Rm, tm = quaternion_matrix(map_odom.transform.rotation), map_odom.transform.translation
        R = Rm @ Ro
        t = Rm @ np.array([to.x, to.y, to.z]) + np.array([tm.x, tm.y, tm.z])
        if not self.store.needs_keyframe(R, t):
            return
        self.last_keyframe = now

        bgr = to_bgr(deserialize_message(raw, Image))
        h, w = bgr.shape[:2]
        fx, cx, kw, kh = self.K
        if (kw, kh) != (w, h):                     # camera_info for another resolution
            fx, cx = fx * w / kw, cx * w / kw
        crops = crop_columns(w, h, self.n_crops)
        t0 = time.perf_counter()
        emb = self.encoder.embed([bgr[:, u0:u1] for u0, u1 in crops])
        scores = relevance(emb, self.pos, self.neg, self.scale, self.temperature)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        depth = None
        draw = self.depth
        if draw is not None and abs(header_of(draw)[0] - stamp) <= self.depth_max_dt:
            dmsg = deserialize_message(draw, Image)
            if dmsg.width == w:
                d = to_array(dmsg)
                depth = depth_profile(d.astype(np.float32) * (0.001 if d.dtype == np.uint16 else 1.0))
        kf = self.store.add(stamp, R, t, fx, cx, crops, scores, depth)
        self.get_logger().debug(
            f'keyframe {kf.id}: S {np.round(scores, 3).tolist()} in {dt_ms:.1f} ms'
            f'{"" if depth is not None else " (no depth)"}')

    def on_frontiers(self, msg):
        frame = msg.header.frame_id or self.map_frame
        out = FrontierRelevance()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.map_frame
        # Frontiers no keyframe has seen get the average view as a neutral prior (observed=false),
        # so an observed frontier only wins or loses against the average scene: the mean raw S,
        # or 0.5 with percentiles. Too few crops for a percentile yet: nothing counts as observed.
        percentile = self.normalization == 'percentile'
        ready = not percentile or self.store.score_count >= self.min_reference
        prior = 0.5 if percentile else self.store.mean_score()
        for goal in msg.goals:
            x, y = goal.position.x, goal.position.y
            if frame != self.map_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(self.map_frame, frame, Time())
                except TransformException:
                    continue
                R = quaternion_matrix(tf.transform.rotation)
                tr = tf.transform.translation
                x, y, _ = R @ np.array([x, y, goal.position.z]) + np.array([tr.x, tr.y, tr.z])
            view = self.store.best_view(x, y) if ready else None
            if view is not None and percentile:
                view = (self.store.percentile(view[0]),) + tuple(view[1:])
            out.positions.append(Point(x=float(x), y=float(y), z=0.0))
            out.relevance.append(float(view[0]) if view else float(prior or 0.0))
            out.observed.append(view is not None)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VlmRelevanceNode()
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
