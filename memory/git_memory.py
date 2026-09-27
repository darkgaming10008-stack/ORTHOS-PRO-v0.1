"""Git-backed versioning for the memory directory.
Auto-commits changes on each save for full history tracking.
Note: SQLite (conversations.db) and ChromaDB are gitignored; this module
versions the remaining memory-dir files (configs, dream_data, etc.).
"""
import subprocess
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
GIT_DIR = MEMORY_DIR / ".git"


def _run_git(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=10,
            cwd=str(MEMORY_DIR),
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def init_git_repo() -> bool:
    """Initialize a git repo in the memory directory if not already present."""
    if GIT_DIR.exists():
        ok, _ = _run_git(["rev-parse", "HEAD"])
        if ok:
            return True
        return _make_initial_commit()

    ok, out = _run_git(["init"])
    if not ok:
        print(f"[GitMemory] Init failed: {out}")
        return False

    # Set local identity for automated commits
    _run_git(["config", "user.email", "memory-system@orthos.local"])
    _run_git(["config", "user.name", "Memory System"])

    gitignore = MEMORY_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "__pycache__/\n*.pyc\n*.npy\n.vector_meta.json\n"
            "conversations.db\n*.db\n*.db-journal\n"
            "dream_data/\n.vector_index.*\n",
            encoding="utf-8",
        )

    ok2 = _make_initial_commit()
    if ok2:
        print("[GitMemory] Initialized git repository")
    else:
        print(f"[GitMemory] Initial commit failed")
    return ok2


def _make_initial_commit() -> bool:
    """Stage and commit all tracked files as initial commit."""
    _run_git(["add", "-A"])
    # Set local identity for automated commits
    _run_git(["config", "user.email", "memory-system@orthos.local"])
    _run_git(["config", "user.name", "Memory System"])
    ok, out = _run_git(["commit", "-m", "Initial memory state"])
    return ok


def commit_memory(message: str = "Auto-save memory state") -> bool:
    """Stage and commit memory files. Returns True if committed."""
    if not GIT_DIR.exists():
        if not init_git_repo():
            return False

    # Stage all memory files
    ok, _ = _run_git(["add", "-A"])
    if not ok:
        return False

    # Check if there's anything to commit
    ok, status = _run_git(["status", "--porcelain"])
    if not ok or not status.strip():
        return False  # nothing changed

    ok, _ = _run_git(["commit", "-m", message])
    if ok:
        print(f"[GitMemory] Committed: {message}")
    return ok


def get_memory_history(n: int = 10) -> list[str]:
    """Return last N commit messages."""
    ok, out = _run_git(["log", f"-{n}", "--oneline"])
    if ok:
        return out.split("\n") if out else []
    return []


def rollback_to(commit_hash: str) -> bool:
    """Restore git-tracked memory files to a specific commit."""
    ok, _ = _run_git(["checkout", commit_hash, "--", "."])
    return ok
