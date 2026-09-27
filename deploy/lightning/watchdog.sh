#!/usr/bin/env bash
# ── Keep-alive watchdog for the Orthos memory service on Lightning ─────────
# Lightning's free tier force-restarts the active CPU Studio every ~4 hours.
# Run this loop (tmux/screen recommended) and the service always comes back:
#
#     tmux new -s memwatch 'bash deploy/lightning/watchdog.sh'
#
# The loop also serves as the machine's "keep session alive" heartbeat.
set -u
cd "$(dirname "$0")/../.."

CHECK_URL="http://127.0.0.1:9876/health"
INTERVAL=60

echo "[watchdog] started $(date) — checking every ${INTERVAL}s"

while true; do
    # 0. Lightning Studios have no systemd — keep tailscaled alive manually.
    if ! pgrep -x tailscaled > /dev/null 2>&1; then
        echo "[watchdog] $(date) tailscaled down — starting (userspace mode)…"
        sudo mkdir -p /var/lib/tailscale
        sudo nohup tailscaled \
            --tun=userspace-networking \
            --state=/var/lib/tailscale/tailscaled.state \
            > /tmp/tailscaled.log 2>&1 &
        sleep 3
        # Re-assert login (no-op when already authenticated)
        sudo tailscale up --hostname=lightning-orthos >/dev/null 2>&1 || true
    fi

    # 1. Is the service healthy?
    if curl -s -m 5 "$CHECK_URL" | grep -q '"status":"ok"'; then
        sleep "$INTERVAL"
        continue
    fi

    echo "[watchdog] $(date) service down — (re)starting…"
    # Kill stale process if any
    if [ -f ~/orthos-memory/service.pid ]; then
        kill "$(cat ~/orthos-memory/service.pid)" 2>/dev/null || true
    fi
    pkill -f "memory_service.py" 2>/dev/null || true
    sleep 2

    # 2. Reinstall deps if the ephemeral overlay was reset by a restart
    #    (pip wheels re-verify quickly; models stay in the persistent cache).
    python -c "import fastapi, uvicorn, sentence_transformers" 2>/dev/null || {
        echo "[watchdog] deps missing — reinstalling…"
        pip install -q fastapi uvicorn pydantic requests numpy \
            sentence-transformers chromadb rank_bm25 || true
    }

    bash deploy/lightning/run_service.sh || true
    sleep "$INTERVAL"
done
