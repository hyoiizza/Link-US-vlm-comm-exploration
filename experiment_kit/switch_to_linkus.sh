#!/usr/bin/env bash
# Move both robots to the experiment Wi-Fi (linkus, power save off) and point config.sh at robot2's new IP.
# Run on robot1 while both are still on a shared network (robot2 reachable at $ROBOT2_HOST).
source ~/experiment_kit/config.sh
R2_MAC=b4:8c:9d:34:f8:61
$SSH "nmcli connection modify linkus 802-11-wireless.powersave 2 connection.autoconnect-priority 10 && \
      (nohup nmcli connection up linkus >/dev/null 2>&1 &) && echo 'robot2: switching to linkus'" || echo "!! robot2 not reachable"
nmcli connection modify linkus 802-11-wireless.powersave 2 connection.autoconnect-priority 10
nmcli connection up linkus >/dev/null && echo "robot1: on linkus"
sleep 8
iw dev wlP1p1s0 link | grep SSID; iw dev wlP1p1s0 get power_save
NET=$(ip -4 -o addr show wlP1p1s0 | awk '{print $4}' | cut -d. -f1-3)
for t in 1 2 3 4 5 6; do
  for i in $(seq 2 254); do (ping -c1 -W1 $NET.$i >/dev/null 2>&1 &); done; sleep 3
  IP=$(ip neigh | grep -i $R2_MAC | grep -v FAILED | awk '{print $1}' | grep "^$NET\." | head -1)
  [ -n "$IP" ] && break
done
[ -z "$IP" ] && { echo "!! robot2 not found on $NET.0/24 (still switching? check its Wi-Fi)"; exit 1; }
sed -i "s/^ROBOT2_HOST=.*/ROBOT2_HOST=\${ROBOT2_HOST:-$IP}   # robot2 on linkus (found by MAC $(date +%F))/" ~/experiment_kit/config.sh
grep -q "$IP" ~/.ssh/known_hosts || ssh-keyscan -T 5 $IP 2>/dev/null >> ~/.ssh/known_hosts
echo "robot2 on linkus: $IP"
ssh -o BatchMode=yes -o ConnectTimeout=5 gitae@$IP 'iw dev wlP1p1s0 link | grep -E "SSID|signal"; iw dev wlP1p1s0 get power_save'
