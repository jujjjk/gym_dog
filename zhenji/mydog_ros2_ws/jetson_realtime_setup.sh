#!/usr/bin/env bash
# One-time Jetson host preparation for the 50 Hz B23500 loop.
#
#   sudo bash jetson_realtime_setup.sh
#
# Measured before this script (5.15.148-tegra, MAXN_SUPER, schedutil): the
# control thread ran at 883 MHz of 1984 MHz while the loop needed 25-30 ms
# per 20 ms cycle in walk. Kernel wake-up latency was under 2 ms, so a
# PREEMPT_RT kernel is not the first lever; clock and scheduling policy are.
#
# What it does (all reversible, nothing touches motor commands):
#   1. rtprio/memlock limits so the controller may request SCHED_FIFO.
#   2. jetson_clocks now and at every boot (locks CPU/GPU/EMC at the
#      maximum of the active nvpmodel mode).
#   3. Keeps the lingzu motor server on CPUs 1-3; start_capture.sh pins the
#      control thread to CPU 5 and the send thread to CPU 4.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo 'run with sudo' >&2
  exit 1
fi
OPERATOR=${SUDO_USER:-jetson}
HOME_DIR=$(getent passwd "$OPERATOR" | cut -d: -f6)

cat > /etc/security/limits.d/99-mydog-realtime.conf <<EOF
${OPERATOR} - rtprio 95
${OPERATOR} - memlock unlimited
EOF
echo "[1/3] rtprio 95 + memlock unlimited for ${OPERATOR} (log in again to apply)"

/usr/bin/jetson_clocks
cat > /etc/systemd/system/mydog-jetson-clocks.service <<'EOF'
[Unit]
Description=Lock Jetson clocks for the 50 Hz motor control loop
After=nvpmodel.service multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable mydog-jetson-clocks.service >/dev/null
echo "[2/3] jetson_clocks applied and enabled at boot"
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  echo "   $(basename "$c") $(cat "$c/cpufreq/scaling_governor") $(cat "$c/cpufreq/scaling_cur_freq") kHz"
done

DROPIN="${HOME_DIR}/.config/systemd/user/lingzu-motor.service.d"
mkdir -p "$DROPIN"
cat > "${DROPIN}/50-cpu-affinity.conf" <<'EOF'
[Service]
CPUAffinity=1 2 3
Nice=-5
EOF
chown -R "$OPERATOR":"$OPERATOR" "${HOME_DIR}/.config/systemd/user/lingzu-motor.service.d"
echo "[3/3] motor server drop-in written: ${DROPIN}/50-cpu-affinity.conf"
echo
echo "Finish as ${OPERATOR}, with the controller stopped:"
echo "  systemctl --user daemon-reload && systemctl --user restart lingzu-motor.service"
echo "  ulimit -r        # must print 95 in a fresh login shell"
echo "Then start_capture.sh live reports realtime_scheduling in /mydog/model23500/status."
