# ☁️ Orthos Memory Service — Cloud Kit (Lightning AI)

**Ek click / ek command me Lightning AI pe memory service start.**

> ⚡ **Lightning Studios VS Code-based hain (Jupyter "Run All" nahi milega) —
> wahan PREFERRED flow ye hai: [`deploy/lightning/README.md`](../deploy/lightning/README.md)
> → VS Code me `Ctrl+Shift+B` (Run Task → Orthos Memory: START).**
> Neeche wala notebook generic Jupyter environments ke liye hai.

---

## 🚀 One-click start (Lightning AI Studio)

### Option 1 — Notebook (recommended)
1. Lightning Studio kholo → `cloud_memory_kit/lightning_memory_service.ipynb` upload/open karo
2. Pehli baar: config cell me `REPO_URL` (aur recommended `TOKEN`) bhar do
3. **Run All** ⏯️ — bas. Ant me **CLOUD URL** print hoga.

### Option 2 — Terminal
```bash
python cloud_memory_kit/start_memory_service.py --repo-url https://github.com/USER/REPO.git --token SECRET123
```

### Ye script khud kya karta hai
| Step | Kaam |
|---|---|
| 1 | Repo clone/pull (fresh Studio pe) |
| 2 | Deps install — sirf **pehli baar** (marker file se skip hota hai) |
| 3 | Tailscale install + login → **stable IP** |
| 4 | Service background me (nohup) — cell band ho to bhi zinda |
| 5 | `/health` wait + **CLOUD URL** + PC config print |

Idempotent: already running ho to restart NAHI karta, sirf health print karta hai.
Dobara chalana safe hai — Studio wapas kholo → Run All → ~10s me ready.

---

## 💻 PC side (ek hi baar)

`config/api_keys.json` me:
```json
{
  "memory_service_url": "http://<TAILSCALE-IP>:9876",
  "memory_service_token": "SECRET123",
  "memory_cloud_only": true
}
```
Orthos restart → done. Cloud memory active (reads + writes dono cloud pe).

---

## 🔧 Troubleshooting

```bash
tail -f memory_service.log                      # live logs
curl http://127.0.0.1:9876/health               # local health
tailscale status                                # PC dikhna chahiye (orthos-pc)
pkill -f memory_service && python cloud_memory_kit/start_memory_service.py   # hard restart
```

- **401 Unauthorized** → PC aur service ka TOKEN same hai? (config vs `--token`)
- **Timeout on PC** → PC pe `tailscale ping <SERVICE-IP>` chalao; dono devices same tailnet me hone chahiye
- **IP badal gaya** → Tailscale login session expire hua hoga; script step 3 me dubara login URL dega. Better: Tailscale admin me **MagicDNS hostname** use karo URL me IP ki jagah.

---

## ❓ Kaggle pe kyun NAHI?

| | Lightning AI | Kaggle |
|---|---|---|
| Storage | ✅ Persistent (index survive) | ❌ Ephemeral (har session re-embed) |
| Limit | Lambe sessions | ❌ 12h hard cap |
| IP | ✅ Stable (Tailscale) | ❌ Har baar naya |

Memory service = **always-on database**. Kaggle LLM inference ke liye best hai,
memory ke liye nahi. (Technically possible — MagicDNS + index rebuild — par
har 12h me pura vector index rebuild + IP change = wahi delay wapas.)
