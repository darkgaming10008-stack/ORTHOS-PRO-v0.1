"""Kaggle provider registration — regression for boot crash
`ValueError: Unknown provider 'kaggle'` (main.py get_default_provider)."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_kaggle_registered_in_provider_registry():
    from core.llm_provider import ProviderRegistry
    assert "kaggle" in ProviderRegistry.list_providers()


def test_get_default_provider_returns_kaggle_instance():
    from core.llm_provider import get_default_provider
    p = get_default_provider()
    assert type(p).__name__ == "KaggleProvider"


def test_kaggle_uses_tunnel_url_and_model():
    from core.llm_provider import get_default_provider
    p = get_default_provider()
    url, model = p.get_settings()
    assert url.startswith("https://"), "kaggle must point at the tunnel URL"
    assert model, "kaggle_model must resolve from config"
    # Must NOT fall back to the localhost ollama defaults
    assert "localhost" not in url and "127.0.0.1" not in url


def test_kaggle_ensure_running_never_launches_local_ollama():
    """ensure_running is overridden — no subprocess 'ollama serve' attempt."""
    import inspect
    from core.llm_provider import KaggleProvider
    src = inspect.getsource(KaggleProvider.ensure_running)
    assert "subprocess" not in src and "Popen" not in src
    # Parent's launcher must not leak into the kaggle path
    assert "launching" not in src
