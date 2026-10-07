#!/usr/bin/env bash
# On robot1 after both robots joined the same Wi-Fi: find robot2 by its MAC, write its IP into config.sh.
R2_MAC=b4:8c:9d:34:f8:61
NET=$(ip -4 -o addr show wlP1p1s0 | awk '{print $4}' | cut -d. -f1-3)
echo "robot1: $(ip -4 -o addr show wlP1p1s0 | awk '{print $4}'), searching $NET.0/24 ..."
for t in 1 2 3 4 5; do
  for i in $(seq 1 254); do (ping -c1 -W1 $NET.$i >/dev/null 2>&1 &); done
  sleep 3
  IP=$(ip neigh | grep -i $R2_MAC | grep -v FAILED | awk '{print $1}' | grep "^$NET\." | head -1)
  [ -n "$IP" ] && break
done
[ -z "$IP" ] && { echo "robot2 not found: is robot2 on the same Wi-Fi?"; exit 1; }
sed -i "s/^ROBOT2_HOST=.*/ROBOT2_HOST=\${ROBOT2_HOST:-$IP}   # found by MAC $(date '+%F %T')/" ~/experiment_kit/config.sh
ssh-keygen -F $IP >/dev/null || ssh-keyscan -T 5 $IP 2>/dev/null >> ~/.ssh/known_hosts
echo "robot2 = $IP (config.sh updated)"
ssh -o BatchMode=yes -o ConnectTimeout=5 gitae@$IP 'echo "ssh ok: $(hostname)"; iw dev wlP1p1s0 link | grep -E "SSID|signal"; iw dev wlP1p1s0 get power_save'
