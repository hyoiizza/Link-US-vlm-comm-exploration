#!/usr/bin/env python3
"""Stop the rover's drive motors directly on the LX-16A bus (no ROS needed).

LX-16A servos in motor mode keep turning at the last commanded speed. If the ROS motor node is
killed before it sends 0 (forced stop, crash), the wheels keep spinning: this sends speed 0 to the
six drive motors, twice. Steering servos are left alone.

    python3 ~/experiment_kit/motors_off.py [/dev/lx16a]
"""
import os
import sys
import termios
import time

DEV = sys.argv[1] if len(sys.argv) > 1 else '/dev/lx16a'
DRIVE_IDS = (2, 3, 5, 7, 8, 10)     # MOTOR_LEFT_FRONT/MIDDLE/BACK, MOTOR_RIGHT_FRONT/MIDDLE/BACK
SERVO_OR_MOTOR_MODE_WRITE = 29


def packet(servo_id, command, params):
    length = 3 + len(params)
    checksum = 255 - ((servo_id + length + command + sum(params)) % 256)
    return bytes([0x55, 0x55, servo_id, length, command, *params, checksum])


def main():
    fd = os.open(DEV, os.O_RDWR | os.O_NOCTTY)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0                                        # iflag: raw
    attrs[1] = 0                                        # oflag: raw
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0                                        # lflag: raw
    attrs[4] = attrs[5] = termios.B115200
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    for _ in range(2):
        for sid in DRIVE_IDS:
            os.write(fd, packet(sid, SERVO_OR_MOTOR_MODE_WRITE, [1, 0, 0, 0]))   # motor mode, speed 0
            time.sleep(0.01)
        time.sleep(0.05)
    termios.tcdrain(fd)
    os.close(fd)
    print(f'{DEV}: speed 0 sent to drive motors {DRIVE_IDS}')


if __name__ == '__main__':
    main()
