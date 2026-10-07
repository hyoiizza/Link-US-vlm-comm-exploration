"""Ping the Wi-Fi gateway (the AP) continuously and publish loss / RTT with the RSSI.

For choosing Q_min: record this with /team/link_quality while driving away from the AP,
then plot packet loss against RSSI (scripts in ~/vlm_check/link_eval or rover_multi docs).

    explore/link_probe  std_msgs/Float32MultiArray, one message per window_s:
        [sent, received, loss (0-1), rtt_mean_ms, rtt_max_ms, rssi_dbm (nan when not associated)]

Runs the system `ping` (no root needed for interval >= 0.2 s) and matches replies to
requests by icmp_seq: a request counts as lost only if no reply came within timeout_s.
("no answer yet" from -O only means "not before the next request": with Wi-Fi power save
replies often take 100-300 ms, and the first version wrongly counted those as losses.)
The SSID / BSSID / gateway are logged at start and whenever they change.
Run:  python3 link_probe.py --ros-args -r __ns:=/robot1 [-p target:=10.0.0.1 -p size:=1000]
      robot-to-robot: -r __node:=link_probe_peer -p target:=<other robot IP> -p topic:=explore/link_probe_peer
"""
import math
import re
import time
import subprocess
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from rover_multi.wifi_monitor import read_level

LABELS = ['sent', 'received', 'loss', 'rtt_mean_ms', 'rtt_max_ms', 'rssi_dbm']
REPLY = re.compile(r'icmp_seq=(\d+).*time=([\d.]+) ms')
SEQ = re.compile(r'icmp_seq=(\d+)')


def association(interface):
    """(ssid, bssid) of the wireless interface from `iw dev <if> link`, ('', '') if not associated."""
    out = subprocess.run(['iw', 'dev', interface, 'link'], capture_output=True, text=True).stdout
    ssid = re.search(r'SSID: (.*)', out)
    bssid = re.search(r'Connected to ([0-9a-f:]{17})', out)
    return (ssid.group(1).strip() if ssid else ''), (bssid.group(1) if bssid else '')


def default_gateway():
    out = subprocess.run(['ip', 'route'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith('default via '):
            return line.split()[2]
    return ''


class LinkProbe(Node):
    def __init__(self):
        super().__init__('link_probe')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.target = p('target', '') or default_gateway()
        self.interval = max(p('interval_s', 0.2), 0.2)
        self.size = p('size', 1000)            # bytes: roughly a mid-size ROS message
        self.interface = p('interface', '') or read_level('')[0]
        self.timeout = p('timeout_s', 1.0)
        if not self.target:
            raise SystemExit('no target and no default gateway')
        self.lock = threading.Lock()
        self.pending = {}        # seq -> monotonic time first seen (request sent, no reply yet)
        self.max_seq = 0
        self.lost = self.received = 0
        self.rtts = []
        self.assoc = None
        # topic: one probe per target, e.g. explore/link_probe (AP) and explore/link_probe_peer (other robot)
        self.pub = self.create_publisher(Float32MultiArray, p('topic', 'explore/link_probe'), 10)
        self.proc = subprocess.Popen(
            ['ping', '-D', '-O', '-n', '-i', str(self.interval), '-s', str(self.size), '-W',
             str(max(1, round(self.timeout))), self.target],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self.read, daemon=True).start()
        self.create_timer(p('window_s', 1.0), self.publish)
        self.get_logger().info(f'Pinging {self.target} every {self.interval}s with {self.size} B '
                               f'({self.interface})')
        self.check_association()

    def check_association(self):
        assoc = association(self.interface) + (default_gateway(),)
        if assoc != self.assoc:
            self.get_logger().info(f'Associated: SSID "{assoc[0]}" BSSID {assoc[1] or "-"} gateway {assoc[2] or "-"}')
            self.assoc = assoc

    def read(self):
        for line in self.proc.stdout:
            m, seq = REPLY.search(line), SEQ.search(line)
            if not seq:
                continue
            n, now = int(seq.group(1)), time.monotonic()
            with self.lock:
                for k in range(self.max_seq + 1, n + 1):      # every request up to this one was sent
                    self.pending[k] = now
                self.max_seq = max(self.max_seq, n)
                if m and n in self.pending:
                    del self.pending[n]
                    self.received += 1
                    self.rtts.append(float(m.group(2)))

    def publish(self):
        now = time.monotonic()
        with self.lock:
            expired = [k for k, t in self.pending.items() if now - t > self.timeout]
            for k in expired:
                del self.pending[k]
            lost, received, rtts = len(expired), self.received, self.rtts
            self.received = 0
            self.rtts = []
        sent = received + lost          # requests resolved in this window (replied or timed out)
        self.check_association()
        _, level = read_level(self.interface)
        loss = 1.0 - received / sent if sent else math.nan
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label=','.join(LABELS), size=len(LABELS), stride=len(LABELS))]
        msg.data = [float(sent), float(received), float(loss),
                    float(sum(rtts) / len(rtts)) if rtts else math.nan, float(max(rtts)) if rtts else math.nan,
                    float(level) if level is not None else math.nan]
        self.pub.publish(msg)

    def destroy_node(self):
        self.proc.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LinkProbe()
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
