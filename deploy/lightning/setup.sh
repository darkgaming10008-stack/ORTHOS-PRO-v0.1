#!/usr/bin/env bash
# ── Orthos Memory Service — Lightning AI Studio one-time setup ──────────────
# Run inside the Lightning AI Studio terminal:  bash deploy/lightning/setup.sh
# Installs deps + pre-downloads models into the persistent (non-ephemeral)
# storage so Studio restarts don't re-download anything.
set -e

echo "════════════════════════════════════════════════════"
echo " Orthos Memory Service — Lightning setup"
echo "════════════════════════════════════════════════════"

# 1. Python deps
echo "→ Installing python deps…"
pip install -q fastapi uvicorn pydantic requests numpy \
    sentence-transformers chromadb rank_bm25 spacy
python -m spacy download en_core_web_sm -q || echo "  (spaCy model optional — NER degraded if missing)"

# 2. Pre-download models into persistent HF cache (~/.cache/huggingface is
#    persistent on Lightning studios; wheel installs are re-run by watchdog).
echo "→ Pre-downloading models…"
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("  MiniLM L6 v2 ✅")
try:
    from sentence_transformers import CrossEncoder
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Reranker ✅")
except Exception as e:
    print(f"  Reranker skipped: {e}")
PY

# 3. Layout for service state on persistent storage
echo "→ Creating persistent state dir…"
mkdir -p ~/orthos-memory/config ~/orthos-memory/logs
ln -sfn "$(pwd)/memory" ~/orthos-memory/memory 2>/dev/null || true

# 4. Interactive token setup
read -rp "Set a MEMORY_TOKEN (press ENTER for random): " TOK
if [ -z "$TOK" ]; then
    TOK=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
fi
echo "MEMORY_TOKEN=$TOK" > ~/orthos-memory/.env
chmod 600 ~/orthos-memory/.env
echo "→ Token saved to ~/orthos-memory/.env (keep it secret; same goes in PC config):"
echo "     memory_service_token = $TOK"

echo
echo "════════════════════════════════════════════════════"
echo " Setup complete ✅"
echo " Next: bash deploy/lightning/run_service.sh"
echo " Then: bash deploy/lightning/watchdog.sh   (keep-alive loop)"
echo "════════════════════════════════════════════════════"
