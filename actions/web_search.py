# web_search.py
# Exa AI search + Ollama LLM summarization + webfetch (URL reading).
# Primary: Exa (built for AI agents)
# Fallback: DuckDuckGo (via ddgs)
import re
import sys
from pathlib import Path

import requests

_HTTP = requests.Session()


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _get_base_dir()

MAX_OUTPUT_CHARS = 50000


def _get_exa_key() -> str | None:
    """Get Exa API key from config."""
    try:
        import json
        cfg_path = BASE_DIR / "config" / "api_keys.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return cfg.get("exa_api_key")
    except Exception:
        pass
    return None


def _has_exa_key() -> bool:
    """Check if Exa API key is configured."""
    key = _get_exa_key()
    return bool(key and key.strip())


def _get_web_search_provider() -> str:
    """Get web search provider from config. Returns 'exa' or 'duckduckgo'."""
    try:
        import json
        cfg_path = BASE_DIR / "config" / "api_keys.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            provider = cfg.get("web_search_provider", "exa")
            return provider if provider in ("exa", "duckduckgo") else "exa"
    except Exception:
        pass
    return "exa"


def _get_exa_default_type() -> str:
    """Get Exa default search type from config. Returns 'fast', 'auto', or 'deep'."""
    try:
        import json
        cfg_path = BASE_DIR / "config" / "api_keys.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            search_type = cfg.get("exa_default_type", "auto")
            return search_type if search_type in ("fast", "auto", "deep") else "auto"
    except Exception:
        pass
    return "auto"


# ─────────────────────────────────────────────────────────────────────────────
# Exa Search
# ─────────────────────────────────────────────────────────────────────────────

def _exa_search(query: str, max_results: int = 8, search_type: str = "auto") -> list[dict]:
    """Search using Exa API (built for AI agents)."""
    from exa_py import Exa

    key = _get_exa_key()
    if not key:
        raise ValueError("No Exa API key")

    exa = Exa(api_key=key)

    # Map search type to Exa type
    type_map = {"fast": "fast", "auto": "auto", "deep": "deep"}
    exa_type = type_map.get(search_type, "auto")

    results = exa.search(
        query,
        num_results=max_results,
        type=exa_type,
    )

    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": getattr(r, "text", "") or getattr(r, "snippet", ""),
        }
        for r in results.results
    ]


def _format_exa(query: str, results: list[dict]) -> str:
    """Format Exa search results."""
    if not results:
        return f"No results found for: {query}"
    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):
            lines.append(f"{i}. {r['title']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# DuckDuckGo Search (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 8) -> list[dict]:
    """Search using DuckDuckGo (fallback when Exa unavailable)."""
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title":   r.get("title",  ""),
                "snippet": r.get("body",   ""),
                "url":     r.get("href",   ""),
            })
    return results


def _format_ddg(query: str, results: list[dict]) -> str:
    """Format DuckDuckGo search results."""
    if not results:
        return f"No results found for: {query}"
    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):   lines.append(f"{i}. {r['title']}")
        if r.get("snippet"): lines.append(f"   {r['snippet']}")
        if r.get("url"):     lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Summarization
# ─────────────────────────────────────────────────────────────────────────────

def _llm_summarize(query: str, raw_results: str, max_context: int = 8000) -> str:
    try:
        from core.llm_client import call_llm_text
        system = (
            "You are Orthos. Summarize web search results clearly and concisely. "
            "Answer the user's query directly. Be factual. Address user as 'sir'."
        )
        prompt = (
            f"User question: {query}\n\n"
            f"Web search results:\n{raw_results[:max_context]}\n\n"
            "Answer the question based on these results:"
        )
        return call_llm_text(prompt, system=system)
    except Exception:
        return raw_results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison Mode
# ─────────────────────────────────────────────────────────────────────────────

def _compare(items: list[str], aspect: str) -> str:
    all_results: dict[str, list] = {}
    for item in items:
        try:
            # Try Exa first, fallback to DuckDuckGo
            if _has_exa_key():
                try:
                    all_results[item] = _exa_search(f"{item} {aspect}", max_results=3)
                except Exception:
                    all_results[item] = _ddg_search(f"{item} {aspect}", max_results=3)
            else:
                all_results[item] = _ddg_search(f"{item} {aspect}", max_results=3)
        except Exception:
            all_results[item] = []

    lines = [f"Comparison — {aspect.upper()}", "─" * 40]
    for item in items:
        lines.append(f"\n▸ {item}")
        for r in all_results.get(item, [])[:2]:
            if r.get("snippet"):
                lines.append(f"  • {r['snippet']}")
    raw = "\n".join(lines)
    return _llm_summarize(f"Compare {', '.join(items)} regarding {aspect}", raw)


# ─────────────────────────────────────────────────────────────────────────────
# HTML Processing
# ─────────────────────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# Web Fetch
# ─────────────────────────────────────────────────────────────────────────────

def _exa_fetch(url: str) -> str | None:
    """Fetch and extract content from URL using Exa Contents API."""
    from exa_py import Exa

    key = _get_exa_key()
    if not key:
        return None

    try:
        exa = Exa(api_key=key)
        result = exa.contents(
            [url],
            highlights=True,
        )

        if result.contents and len(result.contents) > 0:
            content = result.contents[0]
            # Get highlights if available, otherwise full text
            text = ""
            if hasattr(content, 'highlights') and content.highlights:
                text = "\n\n".join(content.highlights)
            elif hasattr(content, 'text') and content.text:
                text = content.text
            return text
    except Exception as e:
        print(f"[WebFetch] Exa contents failed: {e}")

    return None


def _requests_fetch(url: str) -> str:
    """Fetch URL using requests (fallback)."""
    resp = _HTTP.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    raw = resp.text

    if "application/json" in content_type:
        import json
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return raw
    elif "text/html" in content_type or "text/plain" in content_type:
        return _html_to_text(raw)
    else:
        return raw[:2000]


def webfetch(parameters: dict, player=None) -> str:
    """Fetch and extract content from a URL.
    
    Primary: Exa Contents API (token-efficient highlights)
    Fallback: requests + BeautifulSoup
    """
    url = (parameters.get("url") or "").strip()

    if not url:
        return "No URL provided."

    if player:
        player.write_log(f"SYS: 🌐 Fetching webpage — {url[:100]}")

    # Try Exa first
    if _has_exa_key():
        try:
            text = _exa_fetch(url)
            if text:
                header = f"--- Content from {url} (via Exa) ---"
                output = header + "\n" + text
                if len(output) > MAX_OUTPUT_CHARS:
                    output = output[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
                if player:
                    player.write_log(f"SYS: ✅ Exa fetch successful")
                return output
        except Exception as e:
            print(f"[WebFetch] Exa fetch failed, using fallback: {e}")

    # Fallback to requests
    try:
        text = _requests_fetch(url)
        text = text.strip()
        if not text:
            return "Page appears to be empty or requires JavaScript."

        header = f"--- Content from {url} ---"
        output = header + "\n" + text
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return output
    except Exception as e:
        return f"Failed to fetch URL: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Web Search
# ─────────────────────────────────────────────────────────────────────────────

def web_search(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """Search the web for information.
    
    Primary: Exa (built for AI agents)
    Fallback: DuckDuckGo
    """
    params = parameters or {}
    query  = params.get("query", "").strip()
    mode   = params.get("mode",  "search").lower().strip()
    items  = params.get("items", [])
    aspect = params.get("aspect", "general").strip() or "general"
    search_type = params.get("type", "auto").lower().strip()
    num_results  = int(params.get("numResults", 0) or 0)

    if not query and not items:
        return "Please provide a search query, sir."

    if items and mode != "compare":
        mode = "compare"

    # Map type -> result count
    type_map = {"fast": 3, "auto": 8, "deep": 20}
    if num_results > 0:
        max_res = min(num_results, 50)
    else:
        max_res = type_map.get(search_type, 8)

    if player:
        provider = _get_web_search_provider()
        if provider == "exa" and _has_exa_key():
            search_engine = "Exa"
        else:
            search_engine = "DuckDuckGo"
        player.write_log(f"[Search] {query or ', '.join(items)}  engine={search_engine} type={search_type}")

    try:
        if mode == "compare" and items:
            return _compare(items, aspect)

        provider = _get_web_search_provider()

        # Use provider setting
        if provider == "exa" and _has_exa_key():
            # Use Exa with DuckDuckGo fallback
            try:
                results = _exa_search(query, max_results=max_res, search_type=search_type)
                raw = _format_exa(query, results)
                print(f"[WebSearch] Exa: {len(results)} result(s).")
            except Exception as e:
                print(f"[WebSearch] Exa failed ({e}), falling back to DuckDuckGo")
                results = _ddg_search(query, max_results=max_res)
                raw = _format_ddg(query, results)
                print(f"[WebSearch] DDG: {len(results)} result(s).")
        else:
            # Use DuckDuckGo ONLY (no Exa)
            results = _ddg_search(query, max_results=max_res)
            raw = _format_ddg(query, results)
            print(f"[WebSearch] DDG: {len(results)} result(s).")

        # fast mode: return raw results directly, no LLM overhead
        if search_type == "fast":
            return raw

        # auto / deep: use LLM to summarize
        ctx = 12000 if search_type == "deep" else 8000
        return _llm_summarize(query, raw, max_context=ctx)

    except Exception as e:
        print(f"[WebSearch] Failed: {e}")
        return f"Search failed, sir: {e}"
