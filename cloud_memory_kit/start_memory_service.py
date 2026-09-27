#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════
 ORTHOS MEMORY SERVICE — Lightning AI one-click starter
════════════════════════════════════════════════════════════════════
 Ek hi command (ya notebook me ek Run All) me:
   1. Repo code sync (clone/pull)
   2. Dependencies install
   3. Tailscale check/setup (stable private IP ke liye)
   4. Memory service background me start (nohup — cell khatam hone
      ke baad bhi chalti rehti hai)
   5. /health wait + PC-side config summary print

 Usage (Lightning terminal ya notebook):
   %run cloud_memory_kit/start_memory_service.py --token SECRET
   python cloud_memory_kit/start_memory_service.py --repo-url https://github.com/USER/REPO --token SECRET

 Idempotent hai — dobara chalao to already-running service detect
 karke sirf health + summary print karega.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

PORT = 9876
HEALTH_TIMEOUT_S = 240          # model load first boot pe slow ho sakta hai
PIP_PKGS = ("fastapi uvicorn pydantic numpy requests "
            "sentence-transformers chromadb spacy msgpack sentencepiece")


def _sh(cmd: str, cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                          text=True, capture_output=True)


def _pick_repo_dir() -> str:
    """Lightning Studio ka default workspace, warna current dir."""
    for cand in ("/teamspace/studios/this_studio", "/content", os.path.expanduser("~")):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


def _repo_ready(repo_dir: str) -> bool:
    return os.path.isfile(os.path.join(repo_dir, "memory", "memory_service.py"))


def step_repo(repo_dir: str, repo_url: str) -> str:
    """Code sync: fresh clone ya pull. Returns repo path."""
    if _repo_ready(repo_dir):
        print(f"[1/5] Repo mil gaya: {repo_dir} — pull karke latest karta hoon…")
        _sh("git pull --ff-only", cwd=repo_dir, check=False)
    else:
        if not repo_url:
            # Repo already present as current folder? (user ne repo ke andar se chalaya)
            if _repo_ready(os.getcwd()):
                repo_dir = os.getcwd()
                print(f"[1/5] Current folder hi repo hai: {repo_dir}")
                return repo_dir
            raise SystemExit(
                "[1/5] Repo nahi mila. --repo-url https://github.com/USER/REPO do, "
                "ya is script ko repo ke andar se chalao."
            )
        print(f"[1/5] Cloning {repo_url} → {repo_dir} …")
        _sh(f"git clone {repo_url} {repo_dir}", check=False)
        if not _repo_ready(repo_dir):
            raise SystemExit(f"[1/5] Clone ke baad bhi repo ready nahi hua: {repo_dir}")
    return repo_dir


def step_deps(repo_dir: str) -> None:
    marker = os.path.join(repo_dir, ".memory_service_deps_installed")
    if os.path.exists(marker):
        print("[2/5] Dependencies pehle se installed (marker mila) — skip.")
        return
    print("[2/5] Installing dependencies (2-4 min, first time)…")
    _sh(f"{sys.executable} -m pip install -q {PIP_PKGS}", check=False)
    # spaCy NER model — fail hone par service phir bhi chalegi (NER optional)
    _sh(f"{sys.executable} -m spacy download en_core_web_sm -q", check=False)
    try:
        open(marker, "w").close()
    except Exception:
        pass
    print("[2/5] Dependencies done.")


def step_tailscale() -> None:
    """Tailscale = stable IP. Har restart pe naya IP nahi milega."""
    if not shutil.which("tailscale"):
        print("[3/5] Tailscale install kar raha hoon (stable IP ke liye)…")
        _sh("curl -fsSL https://tailscale.com/install.sh | sh", check=False)
    if shutil.which("tailscale"):
        st = _sh("tailscale ip -4", check=False)
        if st.returncode == 0 and st.stdout.strip():
            print(f"[3/5] Tailscale OK — IP: {st.stdout.strip().splitlines()[0]}")
        else:
            print("[3/5] Tailscale login chahiye — neeche AUTH URL par click kar:")
            r = _sh("tailscale up --hostname=orthos-memory", check=False)
            print(r.stdout or r.stderr or "   (tailscale up — output upar dekho)")
    else:
        print("[3/5] WARNING: Tailscale nahi hua install — VM IP har restart pe badlega!")


def _service_pid() -> int | None:
    r = _sh("pgrep -f 'memory.service|memory_service' -d ' '", check=False)
    if r.returncode == 0 and r.stdout.strip():
        try:
            return int(r.stdout.strip().split()[0])
        except ValueError:
            pass
    return None


def _health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def step_start(repo_dir: str) -> None:
    pid = _service_pid()
    if pid and _health_ok():
        print(f"[4/5] Service ALREADY running (pid {pid}) — restart nahi kar raha.")
        return
    if pid:
        print(f"[4/5] Stale process (pid {pid}) — kill karke fresh start…")
        _sh(f"kill {pid}", check=False)
        time.sleep(2)
    print("[4/5] Starting memory service (background, nohup)…")
    env = dict(os.environ)
    if args_token := os.environ.get("MEMORY_TOKEN", ""):
        env["MEMORY_TOKEN"] = args_token
    env.setdefault("MEMORY_HOST", "0.0.0.0")
    env.setdefault("MEMORY_PORT", str(PORT))
    log = os.path.join(repo_dir, "memory_service.log")
    with open(log, "w") as lf:
        subprocess.Popen(
            [sys.executable, "-m", "memory.memory_service"],
            cwd=repo_dir, env=env, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,     # terminal/cell band ho to bhi zinda rahe
        )
    print(f"[4/5] Waiting for /health (first load 20-60s)…")


def step_wait_and_report(repo_dir: str) -> None:
    t0 = time.time()
    while time.time() - t0 < HEALTH_TIMEOUT_S:
        if _health_ok():
            break
        time.sleep(3)
    ts_ip = ""
    if shutil.which("tailscale"):
        r = _sh("tailscale ip -4", check=False)
        if r.returncode == 0 and r.stdout.strip():
            ts_ip = r.stdout.strip().splitlines()[0]
    if _health_ok():
        url = f"http://{ts_ip}:{PORT}" if ts_ip else f"http://127.0.0.1:{PORT}"
        print(
            "\n════════ ORTHOS MEMORY SERVICE READY ════════\n"
            f"  Cloud URL : {url}   ← ye PC ke config/api_keys.json me\n"
            "              'memory_service_url' me paste karo\n"
            f"  Token     : {'SET (Bearer auth on)' if os.environ.get('MEMORY_TOKEN') else 'OFF (token set karna recommended)'}\n"
            f"  Health    : curl {url}/health\n"
            f"  Logs      : tail -f {os.path.join(repo_dir, 'memory_service.log')}\n"
            "══════════════════════════════════════════════\n"
            "  PC side: Orthos restart karo — bas. Cloud memory active.\n"
        )
    else:
        print(
            "\n[!] Service health timeout — logs dekho:\n"
            f"    tail -n 50 {os.path.join(repo_dir, 'memory_service.log')}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Orthos memory service one-click starter")
    ap.add_argument("--repo-url", default=os.environ.get("ORTHOS_REPO_URL", ""),
                    help="GitHub repo URL (sirf fresh machine pe chahiye)")
    ap.add_argument("--repo-dir", default="", help="Repo location (default: auto-detect)")
    ap.add_argument("--token", default=os.environ.get("MEMORY_TOKEN", ""),
                    help="MEMORY_TOKEN (Bearer auth). Empty = auth off.")
    a = ap.parse_args()

    repo_dir = a.repo_dir or _pick_repo_dir()
    if a.token:
        os.environ["MEMORY_TOKEN"] = a.token
    elif not sys.stdin.isatty():
        os.environ.setdefault("MEMORY_TOKEN", "")

    print("── Orthos Memory Service · Lightning one-click ──")
    repo_dir = step_repo(repo_dir, a.repo_url)
    step_deps(repo_dir)
    step_tailscale()
    step_start(repo_dir)
    step_wait_and_report(repo_dir)


if __name__ == "__main__":
    main()
