# 🟢 BEGINNER GUIDE — Cloud Memory (Lightning AI)

> Simple bhasha me: Orthos ki **memory** Lightning AI ke computer pe chalti hai
> (tera i3 slow hai uske liye). Lightning ka computer ghanto me band ho jaata hai,
> isliye script sab kuch khud start karti hai. **Tujhe sirf Ctrl+Shift+B dabana hai.**

---

## 🟢 PART A — Memory service START karna (Lightning pe)

### Step 1 — Lightning kholo
1. Browser me `lightning.ai` → login → apna **Studio** open karo

### Step 2 — Repo check karo
1. VS Code me khul jayega. **Terminal → New Terminal** (ya `Ctrl + ~`)
2. Type karo: `ls` aur Enter
3. Agar `main.py`, `memory` dikh raha hai → Step 3 pe jao
4. Agar nahi dikh raha (naya Studio) → ye chalao:
   ```
   git clone https://github.com/TERA-USERNAME/TERA-REPO.git orthos
   cd orthos
   ```

### Step 3 — EK CLICK START ⭐
- **`Ctrl + Shift + B`** dabao. Bas.
- Script khud: deps install → model download → Tailscale → service start → watchdog
- ⏳ Pehli baar 5-10 min (download). Uske baad ~30 second

### Step 4 — Output padho (end me)
```
Cloud URL : http://100.x.y.z:9876   ← note karo
Token     : abcdXYZ...              ← ye bhi
```
📸 Screenshot le lo!

### Step 5 — Agar Tailscale login maange (sirf pehli baar)
- Output me link dikhega `https://login.tailscale.com/...` → click karo → Google login
- Wapas Lightning me **`Ctrl + Shift + B`** dobara dabao

---

## 🟡 PART B — Token copy (sirf ek baar)

1. `Ctrl + Shift + P` → type karo `Run Task` → Enter
2. **`Orthos Memory: TOKEN dikhao`** select karo
3. `MEMORY_TOKEN=` ke **baad** wala hissa select karke **Ctrl+C**

---

## 🔵 PART C — PC ko cloud se JODO (sirf ek baar)

1. Orthos folder me **`start_memory_service_cloud.bat`** double-click
2. **`2`** (CLOUD) → Enter
3. **URL paste** karo (right-click = paste) → Enter
4. **Token paste** karo → Enter
5. `[OK]` dikhe → **Orthos restart**

---

## 🟣 PART D — Check karo

- PC cmd me: `curl http://100.x.y.z:9876/health` → `{"status":"ok"}` aana chahiye
- Orthos me msg bhejo → `[Timing]` console me fast dikhna chahiye

---

## 🔴 PART E — Problems + Fixes

| Problem | Fix |
|---|---|
| Ctrl+Shift+B se kuch nahi | VS Code me repo folder khula hona chahiye (left panel me files) |
| Connect nahi ho raha | Lightning Studio band ho gaya — kholo → Ctrl+Shift+B |
| IP badal gaya | PART C dobara (naye IP ke saath) |
| 401 Unauthorized | Token mismatch — PART B se copy → PART C |
| Kuch bhi ajeeb | Ctrl+Shift+P → Run Task → **RESTART** |

---

## 💡 Ek line ka niyam
**Lightning kholo → Ctrl+Shift+B → screenshot me URL/token → PC pe .bat se paste → done.**

Purane tarike (manual commands) ke liye: `deploy/lightning/README.md`
