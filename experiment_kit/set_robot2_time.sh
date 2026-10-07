#!/usr/bin/env bash
# robot2 has no RTC battery: after a reboot without internet its clock is 1970. Copy robot1's clock.
# Asks for robot2's sudo password.
source ~/experiment_kit/config.sh
ssh -t ${ROBOT2_USER}@${ROBOT2_HOST} "sudo date -s @$(date +%s.%N) >/dev/null && date"
python3 -c "import time,subprocess;t0=time.time();r=float(subprocess.check_output('$SSH date +%s.%N',shell=True));t1=time.time();print('offset robot2-robot1: %.3f s' % (r-(t0+t1)/2))"
