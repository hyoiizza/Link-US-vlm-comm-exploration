import os
import time

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from rover_object.image_conv import from_bgr, to_bgr
from rover_object.trt_detector import TrtYoloDetector

CLASS_NAMES = ['person', 'fire']
COLORS = [(0, 200, 0), (0, 0, 255)]


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.declare_parameter('engine_path', '')
        self.declare_parameter('image_topic', 'camera/color/image_raw')
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('publish_debug_image', True)
        # Detections per second at most. Every 15 fps frame cost too much CPU next to SLAM + VLM
        # + Nav2 on the Jetson (load 40, 2026-10-01); frames in between are dropped undecoded.
        self.declare_parameter('max_rate_hz', 3.0)

        engine_path = self.get_parameter('engine_path').value
        if not engine_path:
            engine_path = os.path.join(
                get_package_share_directory('rover_object'), 'models', 'optimized', 'best_fp16.engine')
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f'TensorRT engine not found: {engine_path} (run scripts/build_engine.sh first)')

        self.detector = TrtYoloDetector(
            engine_path,
            conf_thres=self.get_parameter('conf_threshold').value,
            iou_thres=self.get_parameter('iou_threshold').value)
        self.get_logger().info(f'Loaded TensorRT engine: {engine_path}')

        self.publish_debug = self.get_parameter('publish_debug_image').value
        self.det_pub = self.create_publisher(Detection2DArray, '~/detections', 10)
        self.dbg_pub = self.create_publisher(Image, '~/debug_image', 1) if self.publish_debug else None
        rate = self.get_parameter('max_rate_hz').value
        self.period = 1.0 / rate if rate > 0 else 0.0
        self.last = 0.0
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.image_cb, qos_profile_sensor_data, raw=True)

    def image_cb(self, raw):
        now = time.monotonic()
        if now - self.last < self.period:
            return
        self.last = now
        msg = deserialize_message(raw, Image)
        frame = to_bgr(msg)
        t0 = time.perf_counter()
        boxes, scores, class_ids = self.detector.infer(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        out = Detection2DArray()
        out.header = msg.header
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids):
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = float((x1 + x2) / 2)
            det.bbox.center.position.y = float((y1 + y2) / 2)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)
            out.detections.append(det)
        self.det_pub.publish(out)

        if self.dbg_pub is not None and self.dbg_pub.get_subscription_count() > 0:
            for (x1, y1, x2, y2), score, cid in zip(boxes.astype(int), scores, class_ids):
                color = COLORS[cid % len(COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{CLASS_NAMES[cid]} {score:.2f}', (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f'{dt_ms:.1f} ms', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            dbg = from_bgr(frame)
            dbg.header = msg.header
            self.dbg_pub.publish(dbg)

    def destroy_node(self):
        self.detector.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
