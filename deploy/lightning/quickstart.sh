#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Orthos Memory Service — Lightning AI ALL-IN-ONE (VS Code friendly)
# ═══════════════════════════════════════════════════════════════════
# Ek command / ek VS Code task (Ctrl+Shift+B) me sab:
#   deps → models → tailscale → service → watchdog → status report
#
# Usage:
#   bash deploy/lightning/quickstart.sh start     # sab kuch (default)
#   bash deploy/lightning/quickstart.sh status    # sirf health + URL
#   bash deploy/lightning/quickstart.sh token     # token dikhao
#   bash deploy/lightning/quickstart.sh stop      # service + watchdog band
#   bash deploy/lightning/quickstart.sh restart   # stop + start
#
# Non-interactive hai: token missing ho to KHUD random generate karke
# ~/orthos-memory/.env me save karta hai (setup.sh ki interactive prompt
# ki zaroorat nahi). Watchdog tmux ke bina nohup background me chalta hai.
set -u
cd "$(dirname "$0")/../.."          # project root
STATE_DIR="$HOME/orthos-memory"
ENV_FILE="$STATE_DIR/.env"
LOG_DIR="$STATE_DIR/logs"
WATCHDOG_PID_FILE="$STATE_DIR/watchdog.pid"
HEALTH_URL="http://127.0.0.1:9876/health"

mkdir -p "$STATE_DIR" "$LOG_DIR"

# ── token: env > .env file > auto-generate ──────────────────────────
load_token() {
    if [ -f "$ENV_FILE" ] && grep -q '^MEMORY_TOKEN=' "$ENV_FILE"; then
        # shellcheck disable=SC1090
        set -a; . "$ENV_FILE"; set +a
    fi
    if [ -z "${MEMORY_TOKEN:-}" ]; then
        MEMORY_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))" 2>/dev/null \
            || openssl rand -hex 16 || echo "orthos-$(date +%s)")
        echo "MEMORY_TOKEN=$MEMORY_TOKEN" > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        echo "[token] naya token generate kiya → $ENV_FILE"
    fi
    export MEMORY_TOKEN
}

# ── deps: marker-based, first run ke baad skip ──────────────────────
ensure_deps() {
    if python -c "import fastapi, uvicorn, pydantic, requests, numpy, sentence_transformers, chromadb, msgpack, sentencepiece" 2>/dev/null; then
        echo "[deps] OK (already installed)"
        return
    fi
    echo "[deps] installing (2-4 min, first run)…"
    pip install -q fastapi uvicorn pydantic requests numpy \
        sentence-transformers chromadb spacy msgpack sentencepiece || true
    python -m spacy download en_core_web_sm -q 2>/dev/null || \
        echo "[deps] spaCy model optional — NER degraded (service phir bhi chalegi)"
    echo "[deps] done"
}

# ── models: persistent HF cache me pre-download (restart pe skip) ───
ensure_models() {
    if python - <<'PY' 2>/dev/null
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
assert m is not None
PY
    then
        echo "[models] MiniLM cached ✅"
        return
    fi
    echo "[models] pre-downloading MiniLM (first run only)…"
    python - <<'PY' || true
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("[models] MiniLM ready ✅")
PY
}

# ── tailscale: stable IP (best-effort, passwordless sudo only) ──────
ensure_tailscale() {
    if command -v tailscale >/dev/null 2>&1 && tailscale ip -4 >/dev/null 2>&1; then
        TS_IP=$(tailscale ip -4 2>/dev/null | head -1)
        echo "[tailscale] OK — IP: $TS_IP"
        return
    fi
    echo "[tailscale] setup…
"
    if ! command -v tailscale >/dev/null 2>&1; then
        curl -fsSL https://tailscale.com/install.sh | sh >/dev/null 2>&1 || true
    fi
    if command -v tailscale >/dev/null 2>&1; then
        # Lightning containers root-ke-paas hote hain; sudo -n = sirf passwordless
        SUDO=""; [ "$(id -u)" != "0" ] && SUDO="sudo -n"
        $SUDO tailscaled --tun=userspace-networking \
            --state="$STATE_DIR/tailscaled.state" >/tmp/tailscaled.log 2>&1 &
        sleep 3
        $SUDO tailscale up --hostname=lightning-orthos 2>&1 | grep -o 'https://login.tailscale.com[^ ]*' && \
            echo "[tailscale] ⚠ UPAR WALE URL PAR LOGIN KARO (ek baar), phir 'start' dobara chalao" || true
        TS_IP=$(tailscale ip -4 2>/dev/null | head -1)
        [ -n "${TS_IP:-}" ] && echo "[tailscale] IP: $TS_IP" || echo "[tailscale] ⚠ IP nahi mila — login ke baad dobara chalao"
    else
        echo "[tailscale] ⚠ install fail — VM public IP hi use hoga (restart pe badlega)"
    fi
}

# ── service: nohup background ───────────────────────────────────────
start_service() {
    if curl -s -m 3 "$HEALTH_URL" 2>/dev/null | grep -q '"ok"'; then
        echo "[service] already running ✅"
        return
    fi
    pkill -f "memory_service" 2>/dev/null || true
    sleep 1
    export MEMORY_HOST=0.0.0.0 MEMORY_PORT=9876
    nohup python memory/memory_service.py > "$LOG_DIR/service.log" 2>&1 &
    echo $! > "$STATE_DIR/service.pid"
    echo "[service] starting (pid $(cat "$STATE_DIR/service.pid")) — model load 20-60s…"
}

wait_health() {
    local i=0
    while [ $i -lt 60 ]; do
        if curl -s -m 3 "$HEALTH_URL" 2>/dev/null | grep -q '"ok"'; then
            echo "[service] healthy ✅"
            return 0
        fi
        sleep 3; i=$((i+1))
    done
    echo "[service] ⚠ health timeout — log: tail -n 40 $LOG_DIR/service.log"
    return 1
}

# ── watchdog: nohup background loop (tmux ki zaroorat nahi) ─────────
start_watchdog() {
    if [ -f "$WATCHDOG_PID_FILE" ] && kill -0 "$(cat "$WATCHDOG_PID_FILE")" 2>/dev/null; then
        echo "[watchdog] already running (pid $(cat "$WATCHDOG_PID_FILE")) ✅"
        return
    fi
    # stale lock cleanup
    [ -f "$WATCHDOG_PID_FILE" ] && rm -f "$WATCHDOG_PID_FILE"
    nohup bash -c '
        while true; do
            curl -s -m 5 http://127.0.0.1:9876/health 2>/dev/null | grep -q "\"ok\"" || {
                bash deploy/lightning/quickstart.sh restart >/dev/null 2>&1 || true
            }
            sleep 60
        done
    ' > "$LOG_DIR/watchdog.log" 2>&1 &
    echo $! > "$WATCHDOG_PID_FILE"
    echo "[watchdog] started (pid $(cat "$WATCHDOG_PID_FILE")) — har 60s health check"
}

stop_service_only() {
    pkill -f "memory_service" 2>/dev/null && echo "[service] stopped" || echo "[service] not running"
}

stop_all() {
    [ -f "$WATCHDOG_PID_FILE" ] && kill "$(cat "$WATCHDOG_PID_FILE")" 2>/dev/null && rm -f "$WATCHDOG_PID_FILE" && echo "[watchdog] stopped"
    stop_service_only
}

# ── status report ────────────────────────────────────────────────────
report() {
    local ts_ip="" ts_url="(tailscale nahi mila)"
    command -v tailscale >/dev/null 2>&1 && ts_ip=$(tailscale ip -4 2>/dev/null | head -1)
    [ -n "$ts_ip" ] && ts_url="http://$ts_ip:9876"
    local health="(down)"
    curl -s -m 3 "$HEALTH_URL" 2>/dev/null | grep -q '"ok"' && health="UP ✅"
    echo ""
    echo "════════ ORTHOS MEMORY SERVICE ════════"
    echo "  Health    : $health"
    echo "  Cloud URL : $ts_url   ← PC config: memory_service_url"
    echo "  Token     : $(grep '^MEMORY_TOKEN=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 || echo '(n/a)')"
    echo "  Logs      : $LOG_DIR/service.log"
    echo "────────────────────────────────────────"
    echo "  PC side: start_memory_service_cloud.bat → [2] CLOUD → URL + token paste"
    echo "═══════════════════════════════════════"
}

# ── main dispatch ────────────────────────────────────────────────────
case "${1:-start}" in
    start)
        load_token; ensure_deps; ensure_models; ensure_tailscale
        start_service; wait_health; start_watchdog; report
        ;;
    # restart: SIRF service band karo — watchdog ko mat maro (watchdog khud
    # yehi command use karta hai; warna khud ko kill karke mari jaata).
    restart)  stop_service_only; sleep 2; "$0" start ;;
    stop)     stop_all ;;
    status)   report ;;
    token)    load_token; echo "MEMORY_TOKEN=$MEMORY_TOKEN" ;;
    *) echo "Usage: $0 {start|stop|restart|status|token}"; exit 1 ;;
esac
