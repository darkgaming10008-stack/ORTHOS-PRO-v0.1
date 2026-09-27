# Orthos Cloud Memory on Lightning AI (free tier)

Your i3 PC can't run MiniLM + cross-encoder reranker fast. This kit moves the
**memory service** (embeddings, NER, reranker, ChromaDB, BM25) to a free
Lightning AI Studio (4 vCPU, 16 GB RAM) while Orthos keeps running locally.
Expected memory-stage latency: **~1–3s → ~120–300ms** per turn.

```
Orthos (i3 PC) ──Tailscale──► Lightning Studio
   │                            └─ memory_service.py :9876
   │                               (MiniLM + NER + reranker + ChromaDB + BM25)
   └─► Groq/Gemini/Kaggle LLM (unchanged)
```

## What stays local vs. goes cloud

| Component | Where | Why |
|---|---|---|
| conversations.db (SQLite) | **PC** | SQLite is instant even on i3 |
| Vector index (ChromaDB/BM25/FAISS) | **Lightning** | embed+search needs CPU muscle |
| MiniLM, reranker, spaCy NER | **Lightning** | models load once, reused for every call |
| LLM | **Cloud (unchanged)** | Groq/Gemini/Kaggle |

## Setup (one time, ~15 min)

### ⚡ VS Code one-click flow (Lightning Studios VS Code-based hain — Jupyter "Run All" nahi)

1. Repo Studio me clone/upload karo → VS Code open karo
2. **`Ctrl+Shift+B`** (ya Terminal → Run Task → **Orthos Memory: START**) — bas.
   Ye ek task me karta hai: deps → model pre-cache → Tailscale → service start →
   background watchdog (auto-restart) → status report (cloud URL + token).
   - Token missing ho to script khud generate karke `~/orthos-memory/.env` me save karta hai
     — phir **Run Task → TOKEN dikhao** se copy karke PC me daal do.
   - Tailscale pehli baar login URL de sakta hai — us par click karke login karo, phir START dobara.
3. PC side: `start_memory_service_cloud.bat` → `[2] CLOUD` → URL + token.

Other tasks: STATUS · TOKEN dikhao · RESTART · STOP · LIVE LOGS (`.vscode/tasks.json`).
CLI equivalent: `bash deploy/lightning/quickstart.sh {start|status|token|restart|stop}`.

### 🔧 Manual flow (purana, step-by-step)

#### 1. Lightning Studio
1. lightning.ai → free account → New Studio (CPU, no GPU needed)
2. Upload/copy the Orthos project folder (or `git clone` your repo)
3. Open the terminal and run:
   ```bash
   bash deploy/lightning/setup.sh      # deps + model pre-cache + token
   ```
   **Save the printed token** — you'll paste it on the PC.

### 2. Tailscale inside Studio (stable private IP)
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=lightning-orthos
tailscale ip -4        # ← note this 100.x.y.z IP (stable forever)
```

### 3. Start the service + watchdog
```bash
tmux new -s memwatch 'bash deploy/lightning/watchdog.sh'
```
Watchdog auto-restarts the service — including after Lightning's ~4-hour
free-tier session restarts.

### 4. On the PC — switch Orthos to cloud memory
Double-click **`start_memory_service_cloud.bat`**:
- Choose `[2] CLOUD`
- URL: `http://100.x.y.z:9876` (Tailscale IP from step 2)
- Token: the one from setup.sh
- It saves to `config/api_keys.json` (`memory_service_url`, `memory_service_token`)
  and health-checks immediately. **No Orthos restart needed.**

Back to local anytime: same bat → `[1] LOCAL`.

## Migrate your existing index (one time)

The cloud starts with an EMPTY vector index — your old memories in SQLite are
safe and still served, but semantic search over OLD chats needs the index.
Migrate it once:

1. **On the PC** (Orthos project root):
   ```bat
   python deploy\lightning\export_from_pc.py
   ```
   → creates `orthos-migrate.zip` (conversations.db + chroma_db + FAISS stores)
2. Upload the zip to the Studio (drag into the file panel)
3. **In the Studio**:
   ```bash
   python deploy/lightning/import_index.py ~/orthos-migrate.zip
   bash deploy/lightning/run_service.sh    # restart to pick up the index
   ```

New chats keep indexing on the cloud automatically after this.

## Safety layers
1. **Tailscale** — service is unreachable from the public internet at all
2. **Bearer token** — even inside the tailnet, requests must carry the token
3. **Auto-fallback** — if the cloud is down AND a local service is running,
   `memory_client` transparently falls back to localhost

## Free-tier caveats (honest list)
- Active CPU session limit (~4h) → watchdog + tmux handles the bounce;
  models stay cached in persistent storage, so restart ≈ 20–40s
- Monthly credits are consumed by studio uptime — check your quota page
- Cold studio (slept) → first request waits for boot; watchdog + warm tmux
  session keeps it mostly warm

## Latency cheatsheet
| Stage | i3 local | Lightning (Tailscale) |
|---|---|---|
| Query embed | 100–500ms | 30–60ms |
| Vector + BM25 | 50–300ms | 20–50ms |
| Rerank (20 cand.) | 0.5–2s | 50–150ms |
| NER | 50–200ms | 20–50ms |
| **Memory total/turn** | **~1–3s** | **~120–300ms** |

Combined with a cloud LLM (Groq/Gemini Flash), end-to-end first response
drops from several seconds to well under a second of added memory overhead.
