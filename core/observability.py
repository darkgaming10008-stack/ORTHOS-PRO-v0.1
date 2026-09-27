"""
Observability module for Orthos.

Provides Prometheus metrics, OpenTelemetry tracing, health checks,
and latency percentiles — all optional (graceful import fallback).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

import requests

_HTTP = requests.Session()

# ── Graceful import helpers ────────────────────────────────────────────────

_HAS_PROMETHEUS = False
_HAS_OTEL = False

try:
    import prometheus_client
    _HAS_PROMETHEUS = True
except ImportError:
    prometheus_client = None  # type: ignore[assignment]

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    trace = None
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    RequestsInstrumentor = None  # type: ignore[assignment]


# ── MetricsCollector ───────────────────────────────────────────────────────


class MetricsCollector:
    """Prometheus metric registry for the application.

    All metric attributes are ``None`` when ``prometheus_client`` is not installed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        if not _HAS_PROMETHEUS:
            self.request_count = None
            self.request_duration = None
            self.token_usage = None
            self.tool_call_count = None
            self.memory_usage = None
            self.active_sessions = None
            self.llm_latency = None
            return

        self.request_count = prometheus_client.Counter(
            name="markxl_requests_total",
            documentation="Total number of requests processed",
            labelnames=["source", "tool_name", "status"],
        )
        self.request_duration = prometheus_client.Histogram(
            name="markxl_request_duration_seconds",
            documentation="Request duration in seconds",
            labelnames=["source", "tool_name"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )
        self.token_usage = prometheus_client.Counter(
            name="markxl_token_usage_total",
            documentation="Number of tokens used",
            labelnames=["model", "type"],
        )
        self.tool_call_count = prometheus_client.Counter(
            name="markxl_tool_calls_total",
            documentation="Number of tool calls executed",
            labelnames=["tool_name", "success"],
        )
        self.memory_usage = prometheus_client.Gauge(
            name="markxl_memory_usage_bytes",
            documentation="Current memory usage in bytes",
        )
        self.active_sessions = prometheus_client.Gauge(
            name="markxl_active_sessions",
            documentation="Number of active sessions",
        )
        self.llm_latency = prometheus_client.Histogram(
            name="markxl_llm_latency_seconds",
            documentation="LLM request latency in seconds",
            labelnames=["model", "provider"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
        )

    def inc_request(self, source: str, tool_name: str = "", status: str = "ok") -> None:
        if self.request_count is not None:
            self.request_count.labels(source=source, tool_name=tool_name, status=status).inc()

    def observe_request_duration(self, source: str, tool_name: str, seconds: float) -> None:
        if self.request_duration is not None:
            self.request_duration.labels(source=source, tool_name=tool_name).observe(seconds)

    def inc_token_usage(self, model: str, token_type: str, count: int = 1) -> None:
        if self.token_usage is not None:
            self.token_usage.labels(model=model, type=token_type).inc(count)

    def inc_tool_call(self, tool_name: str, success: bool = True) -> None:
        if self.tool_call_count is not None:
            self.tool_call_count.labels(tool_name=tool_name, success=str(success).lower()).inc()

    def set_memory_usage(self, bytes_: float) -> None:
        if self.memory_usage is not None:
            self.memory_usage.set(bytes_)

    def set_active_sessions(self, count: int) -> None:
        if self.active_sessions is not None:
            self.active_sessions.set(count)

    def observe_llm_latency(self, model: str, provider: str, seconds: float) -> None:
        if self.llm_latency is not None:
            self.llm_latency.labels(model=model, provider=provider).observe(seconds)


# ── Metrics server ─────────────────────────────────────────────────────────


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics server on the given *port*.

    Starts a background thread running ``prometheus_client.start_http_server``.
    Safe to call multiple times — only the first call starts the server.
    """
    if not _HAS_PROMETHEUS:
        print("[observability] prometheus_client not installed — cannot start metrics server")
        return

    if getattr(start_metrics_server, "_started", False):
        return

    def _serve() -> None:
        try:
            prometheus_client.start_http_server(port)
            print(f"[observability] Prometheus metrics server listening on :{port}")
        except Exception as e:
            print(f"[observability] Failed to start metrics server on :{port} — {e}")

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    start_metrics_server._started = True


# ── Timer context manager ──────────────────────────────────────────────────


class Timer:
    """Context manager that measures elapsed wall-clock time.

    Usage::

        with Timer() as t:
            do_something()
        print(f"Took {t.elapsed:.3f}s")

    The elapsed time is also available via ``str(t)`` and ``float(t)``.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def __float__(self) -> float:
        return self._elapsed

    def __str__(self) -> str:
        return f"{self._elapsed:.3f}s"


@contextmanager
def timer():
    """Convenience :func:`~contextlib.contextmanager` variant of :class:`Timer`.

    Usage::

        with timer() as t:
            do_something()
        print(f"Took {t:.3f}s")   # t is a float
    """
    start = time.perf_counter()
    try:
        yield start
    finally:
        pass
    # Yield the elapsed time after the block completes
    yield time.perf_counter() - start


# ── OpenTelemetry tracing ──────────────────────────────────────────────────


def init_tracing(service_name: str = "mark-xl") -> bool:
    """Initialise OpenTelemetry tracing with OTLP HTTP export.

    Auto-instruments the ``requests`` library for HTTP call tracing.
    Returns ``True`` if tracing was successfully initialised, ``False``
    if the optional OpenTelemetry packages are not installed.

    Parameters
    ----------
    service_name : str
        Value for the ``service.name`` resource attribute.
    """
    if not _HAS_OTEL:
        print("[observability] OpenTelemetry packages not installed — tracing disabled")
        return False

    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        RequestsInstrumentor().instrument()

        print(f"[observability] OpenTelemetry initialised (service='{service_name}')")
        return True
    except Exception as e:
        print(f"[observability] OpenTelemetry init failed — {e}")
        return False


# ── HealthCheck ────────────────────────────────────────────────────────────


def _ollama_is_running() -> tuple[bool, str]:
    """Check if the Ollama server is reachable at the default endpoint."""
    url = "http://localhost:11434/api/tags"
    try:
        resp = _HTTP.get(url, timeout=5)
        if resp.status_code == 200:
            return True, "ollama reachable"
        return False, f"ollama returned HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "ollama not reachable (is it running?)"
    except Exception as e:
        return False, str(e)


def _memory_service_health() -> tuple[bool, str]:
    """Check memory service health on its default port."""
    url = "http://localhost:8000/health"
    try:
        resp = _HTTP.get(url, timeout=5)
        if resp.status_code == 200:
            return True, "memory service healthy"
        return False, f"memory service returned HTTP {resp.status_code}"
    except requests.ConnectionError:
        return False, "memory service not reachable"
    except Exception as e:
        return False, str(e)


def _stt_ready() -> tuple[bool, str]:
    """Check STT readiness by importing the STT module.

    This is a lightweight module-level check — it does not verify that
    a model is loaded (that is done at configuration time).
    """
    try:
        import importlib
        importlib.import_module("core.stt")
        return True, "stt module importable"
    except Exception as e:
        return False, str(e)


def _tts_ready() -> tuple[bool, str]:
    """Check TTS readiness by importing the TTS module."""
    try:
        import importlib
        importlib.import_module("core.tts")
        return True, "tts module importable"
    except Exception as e:
        return False, str(e)


class HealthCheck:
    """Registry of health-check callables with a ``run_all()`` runner.

    Each registered check is a ``Callable[[], tuple[bool, str]]`` that
    returns ``(ok, detail)``.

    Usage::

        hc = HealthCheck()
        hc.register_check("my_service", my_check_fn)
        results = hc.run_all()
    """

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[], tuple[bool, str]]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register_check("ollama_is_running", _ollama_is_running)
        self.register_check("memory_service", _memory_service_health)
        self.register_check("stt_ready", _stt_ready)
        self.register_check("tts_ready", _tts_ready)

    def register_check(self, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        """Register a health check under *name*.

        *fn* must return ``(is_healthy: bool, detail: str)``.
        """
        self._checks[name] = fn

    def run_all(self) -> dict[str, dict[str, str]]:
        """Run every registered health check and return the results.

        Returns a dict mapping check names to::

            {"status": "ok" | "fail", "detail": "…"}
        """
        results: dict[str, dict[str, str]] = {}
        for name, fn in self._checks.items():
            try:
                ok, detail = fn()
            except Exception as e:
                ok, detail = False, str(e)
            results[name] = {
                "status": "ok" if ok else "fail",
                "detail": detail,
            }
        return results


# ── LatencyTracker ─────────────────────────────────────────────────────────


class LatencyTracker:
    """Tracks p50 / p95 / p99 latencies per tool name.

    All latencies are stored in-memory.  Call :meth:`record` after each
    tool invocation, then query percentiles with :meth:`summary`.

    Usage::

        tracker = LatencyTracker()
        tracker.record("get_weather", 0.45)
        tracker.record("get_weather", 0.52)
        print(tracker.summary("get_weather"))
        # {"p50": 0.485, "p95": 0.517, "p99": 0.519, "count": 2, "mean": 0.485}
    """

    def __init__(self, max_samples: int = 5000) -> None:
        self._max = max_samples
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = {}

    def record(self, tool_name: str, seconds: float) -> None:
        """Record a latency sample for *tool_name*."""
        with self._lock:
            bucket = self._samples.setdefault(tool_name, [])
            bucket.append(seconds)
            if len(bucket) > self._max:
                bucket.pop(0)

    def summary(self, tool_name: str) -> dict[str, float]:
        """Return percentile summary for *tool_name*.

        Returns keys ``p50``, ``p95``, ``p99``, ``count``, ``mean``.
        All zeros when there are no samples.
        """
        with self._lock:
            raw = self._samples.get(tool_name)
            if not raw:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0, "mean": 0.0}
            s = sorted(raw)
        n = len(s)
        return {
            "p50": s[int(n * 0.50)] if n else 0.0,
            "p95": s[int(n * 0.95)] if n else 0.0,
            "p99": s[int(n * 0.99)] if n else 0.0,
            "count": n,
            "mean": sum(s) / n,
        }

    def all_summaries(self) -> dict[str, dict[str, float]]:
        """Return percentile summaries for every tracked tool."""
        names = list(self._samples.keys())
        return {name: self.summary(name) for name in sorted(names)}

    def clear(self, tool_name: str | None = None) -> None:
        """Clear samples for *tool_name*, or all tools if ``None``."""
        with self._lock:
            if tool_name:
                self._samples.pop(tool_name, None)
            else:
                self._samples.clear()
