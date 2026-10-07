"""sensor_msgs/Image <-> numpy without cv_bridge.

cv_bridge's compiled module is built against NumPy 1.x and crashes on robot2, which has
NumPy 2 (needed by ultralytics). These conversions only reshape the buffer, so they work with
either NumPy and cost nothing extra.
"""
import numpy as np
from sensor_msgs.msg import Image

_DTYPE = {'8UC1': (np.uint8, 1), 'mono8': (np.uint8, 1), '16UC1': (np.uint16, 1), 'mono16': (np.uint16, 1),
          '32FC1': (np.float32, 1), 'bgr8': (np.uint8, 3), 'rgb8': (np.uint8, 3),
          'bgra8': (np.uint8, 4), 'rgba8': (np.uint8, 4)}


def to_array(msg):
    """The image as stored (passthrough): [H, W] or [H, W, C], honouring step and endianness."""
    if msg.encoding not in _DTYPE:
        raise ValueError(f'unsupported image encoding {msg.encoding!r}')
    dtype, ch = _DTYPE[msg.encoding]
    dt = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    row = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, memoryview)) else msg.data,
                        dtype=np.uint8).reshape(msg.height, msg.step)
    a = row[:, :msg.width * ch * dt.itemsize].copy().view(dt).reshape(msg.height, msg.width, ch)
    return a[:, :, 0] if ch == 1 else a


def to_bgr(msg):
    """uint8 BGR [H, W, 3] from bgr8 / rgb8 / bgra8 / rgba8 / mono8."""
    a = to_array(msg)
    if msg.encoding == 'bgr8':
        return a
    if msg.encoding == 'rgb8':
        return np.ascontiguousarray(a[:, :, ::-1])
    if msg.encoding == 'bgra8':
        return np.ascontiguousarray(a[:, :, :3])
    if msg.encoding == 'rgba8':
        return np.ascontiguousarray(a[:, :, 2::-1])
    if a.ndim == 2 and a.dtype == np.uint8:
        return np.repeat(a[:, :, None], 3, axis=2)
    raise ValueError(f'cannot make BGR from {msg.encoding!r}')


def from_bgr(bgr, header=None):
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr, dtype=np.uint8).tobytes()
    return msg
