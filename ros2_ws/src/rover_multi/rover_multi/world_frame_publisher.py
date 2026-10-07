"""Anchor every robot's SLAM map in a shared ``world`` frame.

Publishes world -> <robot>/map for each robot. For now the transform is the
robot's start pose at the shared entrance (parameters, changeable at runtime
with ``ros2 param set``). Map-to-map corrections will later update the same
transform, so consumers never need to change.

/tf_static is used on purpose: it is latched (late joiners get it), has no
timestamp extrapolation issues, and a newer message for the same child frame
replaces the old one, so it can still be updated while running.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class WorldFramePublisher(Node):
    def __init__(self):
        super().__init__('world_frame_publisher')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('robots', ['robot1', 'robot2'])
        self.world_frame = self.get_parameter('world_frame').value
        self.robots = list(self.get_parameter('robots').value)

        # <robot>.initial_pose: [x, y, z, yaw(rad)] of the robot's map origin in the world frame.
        self.poses = {}
        for robot in self.robots:
            pose = self.declare_parameter(f'{robot}.initial_pose', [0.0, 0.0, 0.0, 0.0]).value
            self.poses[robot] = self._validate(robot, pose)

        self.broadcaster = StaticTransformBroadcaster(self)
        self.add_on_set_parameters_callback(self._on_params)
        self._publish()

    def _validate(self, robot, pose):
        pose = [float(v) for v in pose]
        if len(pose) != 4:
            raise ValueError(f'{robot}.initial_pose must be [x, y, z, yaw], got {pose}')
        return pose

    def _on_params(self, params):
        updated = {}
        for p in params:
            robot = p.name[:-len('.initial_pose')] if p.name.endswith('.initial_pose') else None
            if robot not in self.poses:
                continue
            try:
                updated[robot] = self._validate(robot, p.value)
            except (TypeError, ValueError) as e:
                return SetParametersResult(successful=False, reason=str(e))
        if updated:
            self.poses.update(updated)
            self._publish()
        return SetParametersResult(successful=True)

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for robot, (x, y, z, yaw) in self.poses.items():
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.world_frame
            t.child_frame_id = f'{robot}/map'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = yaw_to_quaternion(yaw)
            transforms.append(t)
            self.get_logger().info(
                f'{self.world_frame} -> {robot}/map: x={x:.3f} y={y:.3f} z={z:.3f} '
                f'yaw={math.degrees(yaw):.1f}deg')
        self.broadcaster.sendTransform(transforms)


def main(args=None):
    rclpy.init(args=args)
    node = WorldFramePublisher()
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
