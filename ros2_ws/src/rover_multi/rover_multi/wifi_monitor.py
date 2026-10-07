"""Measure this robot's Wi-Fi link and publish it with the robot's world position.

Reads the signal level of the wireless interface from /proc/net/wireless (no
subprocess, cheap enough for a few Hz on the Jetson) and publishes
rover_msgs/LinkQuality on /team/link_quality for the coordinator's link model
and the experiment metrics.

    interface   "" = the first interface listed in /proc/net/wireless
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from rover_msgs.msg import LinkQuality
from rover_multi import team_interface as team

PROC = '/proc/net/wireless'


def read_level(interface, path=PROC):
    """(interface, level dBm) from /proc/net/wireless, or (interface, None) when not associated."""
    with open(path) as f:
        lines = f.readlines()[2:]
    for line in lines:
        name, _, rest = line.partition(':')
        name = name.strip()
        if interface and name != interface:
            continue
        fields = rest.split()
        # status, link quality, level, noise, ...
        level = float(fields[2].rstrip('.'))
        # A disassociated interface reports 0 or a positive sentinel level.
        return name, (level if level < 0 else None)
    return interface, None


class WifiMonitor(Node):
    def __init__(self):
        super().__init__('wifi_monitor')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.robot = p('robot_name', self.get_namespace().strip('/'))
        self.interface = p('interface', '')
        self.world_frame = p('world_frame', 'world')
        self.base_frame = p('base_frame', f'{self.robot}/base_link')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(LinkQuality, team.LINK_QUALITY, team.LINK_QUALITY_QOS)
        self.create_timer(1.0 / p('rate_hz', 2.0), self.tick)
        self.get_logger().info(f'Wi-Fi monitor for {self.robot} on {self.interface or "first wireless interface"}')

    def tick(self):
        try:
            name, level = read_level(self.interface)
        except OSError as e:
            self.get_logger().warn(f'Cannot read {PROC}: {e}', throttle_duration_sec=10.0)
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, Time())
        except TransformException as e:
            self.get_logger().warn(f'No pose in {self.world_frame}: {e}', throttle_duration_sec=10.0)
            return
        msg = LinkQuality(robot=self.robot, connected=level is not None,
                          rssi_dbm=float(level) if level is not None else -100.0)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        t = tf.transform.translation
        msg.position.x, msg.position.y, msg.position.z = t.x, t.y, t.z
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WifiMonitor()
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
