"""Decoder for rtabmap's RVL-compressed depth images ("DEPTHRVL" + int32 width + int32 height + data).

RVL (A. Wilson, "Fast Lossless Depth Image Compression", 2017): runs of zeros and of non-zero
pixels, each pixel stored as a zig-zag delta to the previous non-zero pixel, all values as
variable-length 3-bit-payload nibbles packed in little-endian 32-bit words (MSB first).
"""
import struct

import numpy as np


def decode_rvl(blob):
    """uint16 depth image [mm] from an rtabmap depth blob, or None if it is not RVL."""
    if blob is None or blob[:8] != b'DEPTHRVL':
        return None
    w, h = struct.unpack_from('<ii', blob, 8)
    words = np.frombuffer(blob, dtype='<u4', offset=16, count=(len(blob) - 16) // 4)
    # every nibble of every word, most significant first
    nib = ((words[:, None] >> np.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=np.uint32)) & 0xF).ravel()
    # variable-length values: 3 payload bits per nibble, high bit = "more nibbles follow"
    ends = np.flatnonzero((nib & 0x8) == 0)
    starts = np.r_[0, ends[:-1] + 1]
    payload = (nib & 0x7).astype(np.int64)
    vals = np.empty(len(ends), dtype=np.int64)
    # value = sum payload[k] << 3*(k - start); most values are 1-3 nibbles, loop over lengths
    lengths = ends - starts + 1
    vals[:] = 0
    for k in range(int(lengths.max()) if len(lengths) else 0):
        m = lengths > k
        vals[m] |= payload[starts[m] + k] << (3 * k)
    n = w * h
    out = np.zeros(n, dtype=np.int64)
    i = pos = 0
    prev = 0
    nv = len(vals)
    while pos < n and i < nv:
        zeros = int(vals[i]); i += 1
        pos += zeros
        if i >= nv:
            break
        nonzeros = int(vals[i]); i += 1
        if nonzeros:
            pos_vals = vals[i:i + nonzeros]; i += nonzeros
            deltas = (pos_vals >> 1) ^ -(pos_vals & 1)
            run = prev + np.cumsum(deltas)
            out[pos:pos + nonzeros] = run
            prev = int(run[-1])
            pos += nonzeros
    return out[:n].reshape(h, w).astype(np.uint16)
