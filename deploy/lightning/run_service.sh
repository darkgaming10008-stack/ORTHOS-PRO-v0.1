#!/usr/bin/env bash
# ── Start the Orthos memory service on Lightning (0.0.0.0 + token) ─────────
# Usage: bash deploy/lightning/run_service.sh
set -e
cd "$(dirname "$0")/../.."   # project root

export MEMORY_HOST=0.0.0.0
export MEMORY_PORT=9876
# Token from setup.sh
if [ -f ~/orthos-memory/.env ]; then
    set -a; source ~/orthos-memory/.env; set +a
fi
export MEMORY_TOKEN="${MEMORY_TOKEN:-}"

mkdir -p ~/orthos-memory/logs
nohup python memory/memory_service.py > ~/orthos-memory/logs/service.log 2>&1 &
echo $! > ~/orthos-memory/service.pid

echo "Memory service starting (pid $(cat ~/orthos-memory/service.pid))…"
sleep 2
# Sanity check
if curl -s -m 5 "http://127.0.0.1:9876/health" | grep -q '"ok"'; then
    echo "✅ Service is up: http://127.0.0.1:9876"
    echo "   (Models still loading in background — /health shows model_loaded)"
else
    echo "⚠ Service not ready yet — check ~/orthos-memory/logs/service.log"
fi
echo "Tailscale IP: $(tailscale ip -4 2>/dev/null || echo 'tailscale not installed — install inside Studio for stable access')"
