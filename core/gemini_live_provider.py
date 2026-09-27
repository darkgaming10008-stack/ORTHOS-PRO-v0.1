"""
Gemini Live API provider — persistent WebSocket session (Mark-L style).

Mirrors the reference implementation (Mark-L's JarvisLive): ONE long-lived
`client.aio.live.connect()` session for the whole conversation instead of a
new session per turn.  The session continuously:

  • streams the mic (16 kHz PCM) to the Live API   — _listen_audio/_send_realtime
  • plays the model's native audio (24 kHz PCM)     — _receive_audio/_play_audio
  • executes tool calls in-session and replies      — _execute_tool
  • reconnects with exponential backoff + session_resumption

The provider keeps Orthos's standard LLMProvider interface.  chat_stream()
bridges one *user turn* (typed text) into the persistent session and yields
sentence/done events built from output transcription.  All audio is Gemini
Live's own voice — events carry "native_audio": True so the caller must NOT
TTS-speak them.
"""

from __future__ import annotations

import asyncio
import os
import queue
import re
import sys as _sys
import threading
import time
import traceback
from typing import Any, Callable, Generator

import sounddevice as sd

from core.llm_provider import LLMProvider, _load_config, _DEFAULTS, _HTTP

# ── Windows console safety ──────────────────────────────────────────────────
# Printing unicode (→, ⚠️, emoji) raises 'charmap' codec errors when stdout/
# stderr is piped or redirected — and that crash kills the Live session
# mid-tool-round (Mark-L main.py:885 has the same hazard; it survives only
# because it runs in a real console).  Keep the detected encoding, never
# raise on unencodable characters.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ── Constants (match Mark-L) ────────────────────────────────────────────────

# This model is available for the configured Gemini API key.  Do not replace
# it with the similarly named ``gemini-live-2.5-flash``: that model is absent
# from this key's v1beta Live catalog and is rejected with WebSocket 1008.
LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')


def _clean_transcript(text: str) -> str:
    """Strip control tags that some models emit inside transcription."""
    return re.sub(r"<[^>]+>", "", text).strip()


_ALL_LIVE_VOICES = [
    "Puck", "Charon", "Kore", "Fenrir", "Leda", "Aoede", "Zephyr",
    "Orus", "Autonoe", "Umbriel", "Erinome", "Laomedeia", "Schedar",
    "Achird", "Sadachbia", "Enceladus", "Algieba", "Algenib", "Achernar",
    "Gacrux", "Zubenelgenubi", "Sadaltager", "Callirrhoe", "Iapetus",
    "Despina", "Rasalgethi", "Alnilam", "Pulcherrima", "Vindemiatrix",
    "Sulafat",
]


class QuotaExceededError(RuntimeError):
    """Raised when the Gemini Live free-tier daily quota is exhausted."""


_QUOTA_MARKERS = (
    "quota",
    "RESOURCE_EXHAUSTED",
    "exceeded your current quota",
    "quotaValue",
    "QuotaFailure",
    "1011",
)


def _is_quota_error(msg: str) -> bool:
    low = msg.lower()
    return any(m in low for m in _QUOTA_MARKERS)


def _is_transient_connection_error(msg: str) -> bool:
    """Network closures are recoverable; avoid treating them as Live bugs."""
    low = msg.lower()
    return any(marker in low for marker in (
        "1006", "abnormal closure", "connection closed", "connection aborted",
        "connection reset", "getaddrinfo", "timed out", "cannot connect",
        "connectionrefused", "network is unreachable",
    ))


class GeminiLiveProvider(LLMProvider):
    """Google Gemini Live — persistent WebSocket session, native audio."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or "https://generativelanguage.googleapis.com").rstrip("/")
        configured_model = model or cfg.get("gemini_live_model") or LIVE_MODEL
        self._model = configured_model.removeprefix("models/")
        self._api_key = api_key or cfg.get("gemini_api_key", "")
        self._voice   = cfg.get("gemini_live_voice", "Charon")

        # Session state
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._quota_error_message = None
        self._resumption_handle = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._connected_event = threading.Event()
        self._audio_in_q: asyncio.Queue | None = None
        self._out_q: asyncio.Queue | None = None
        self._turn_done_event: asyncio.Event | None = None
        self._mic = None
        self._out_stream = None
        self._conn_backoff = 3
        self._last_transcription = ""

        # Turn / speech state
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self._interrupted = False
        # Typed turns use send_client_content.  Pause microphone streaming for
        # those turns because Gemini Live does not reliably handle concurrent
        # client-content and realtime-audio input in one session.
        self._text_turn_active = False
        self._tool_round = False
        self._content_after_tool = False
        self._turn_acc = ""
        self._sent_buf = ""
        # State assignment is atomic under CPython.  An RLock keeps cleanup
        # safe without blocking the receiver while chat_stream waits on q.
        self._turn_lock = threading.RLock()
        self._active_turn_q: queue.Queue | None = None

        # Config injected by main.py before the session connects
        self._system_instruction: str | None = None
        self._tools = None
        self._tool_handler: Callable[[str, dict], str] | None = None
        self._output_cb: Callable[[str], None] | None = None
        self._user_cb: Callable[[str], None] | None = None
        self._muted_getter: Callable[[], bool] | None = None
        self._speaking_cb: Callable[[bool], None] | None = None

    def reconfigure(self, *, model: str | None = None, api_key: str | None = None,
                    url: str | None = None) -> None:
        """Refresh settings on the registry-cached persistent session."""
        changed = False
        if model:
            normalized = model.removeprefix("models/")
            if normalized != self._model:
                self._model, changed = normalized, True
        if api_key is not None and api_key != self._api_key:
            self._api_key, changed = api_key, True
        if url:
            normalized_url = url.rstrip("/")
            if normalized_url != self._url:
                self._url, changed = normalized_url, True
        if changed:
            self._resumption_handle = None
            self._quota_error_message = None
            self._close_current_session()

    def _close_current_session(self) -> None:
        session, loop = self._session, self._loop
        if session is None or loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(session.close(), loop)
        except Exception:
            pass

    # ── Public interface (LLMProvider contract) ────────────────────────────

    def get_settings(self) -> tuple[str, str]:
        return self._url, self._model

    @property
    def session_active(self) -> bool:
        return self._session is not None and self._connected_event.is_set()

    def ensure_running(self) -> bool:
        if not self._api_key:
            return False
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._session_main, daemon=True)
            self._thread.start()
        return True

    def warmup(self, system_prompt: str | None = None) -> bool:
        if system_prompt:
            self.set_system_prompt(system_prompt)
        print(f"[GeminiLive] Warming up '{self._model}'…")
        try:
            from google import genai

            client = genai.Client(
                api_key=self._api_key,
                http_options={"api_version": "v1beta"},
            )
            models = client.models.list()
            for m in models:
                if m.name.removeprefix("models/") == self._model:
                    print(f"[GeminiLive] '{self._model}' ready.")
                    self.ensure_running()
                    return True
            print(f"[GeminiLive] Model '{self._model}' not found in available models.")
            return False
        except Exception as e:
            print(f"[GeminiLive] Warmup failed (non-fatal): {e}")
            return False

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        gen = self.chat_stream(messages, tools, timeout)
        content = ""
        tool_calls = []
        for event in gen:
            if event["type"] in ("done", "cancelled"):
                content = event["content"]
                tool_calls = event["tool_calls"]
        return {"content": content, "tool_calls": tool_calls}

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        if not self.ensure_running():
            yield {
                "type": "done",
                "content": "Gemini Live API key is not configured.",
                "tool_calls": [],
            }
            return
        if tools:
            self.set_tools(tools)
        wait_deadline = time.monotonic() + 30
        while not self._connected_event.is_set():
            if self._quota_error_message:
                # Propagate quota exhaustion as a specific error
                raise QuotaExceededError(self._quota_error_message)
            if time.monotonic() >= wait_deadline:
                raise RuntimeError(
                    "Gemini Live session could not connect — check API key and network."
                )
            time.sleep(0.1)

        user_text = self._build_live_input(messages)
        if not user_text.strip():
            yield {"type": "done", "content": "", "tool_calls": []}
            return

        turn_q: queue.Queue = queue.Queue()
        with self._turn_lock:
            self._active_turn_q = turn_q
            self._turn_acc = ""
            self._interrupted = False
            self._last_transcription = ""  # reset for new turn
            self._text_turn_active = True
            try:
                # Discard silence / mic chunks queued before the typed turn;
                # otherwise they can delay its response by several seconds.
                pause_fut = asyncio.run_coroutine_threadsafe(
                    self._drain_realtime_input(), self._loop
                )
                pause_fut.result(timeout=5)
                fut = asyncio.run_coroutine_threadsafe(
                    self._send_text(user_text),
                    self._loop,
                )
                fut.result(timeout=15)

                deadline = time.monotonic() + timeout
                while True:
                    if cancel_event and cancel_event.is_set():
                        self.interrupt()
                        yield {
                            "type": "cancelled",
                            "content": "",
                            "tool_calls": [],
                        }
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.interrupt()
                        raise RuntimeError("Gemini Live stream timed out.")
                    try:
                        ev = turn_q.get(timeout=min(0.5, remaining))
                    except queue.Empty:
                        if not self._connected_event.is_set():
                            raise RuntimeError(
                                "Gemini Live session lost connection — reconnecting."
                            )
                        continue
                    if ev["type"] == "error":
                        if ev.get("quota"):
                            raise QuotaExceededError(ev["message"])
                        raise RuntimeError(ev["message"])
                    yield ev
                    if ev["type"] in ("done", "cancelled"):
                        return
            except Exception as e:
                raise RuntimeError(f"Gemini Live send failed: {e}")
            finally:
                self._text_turn_active = False
                with self._turn_lock:
                    if self._active_turn_q is turn_q:
                        self._active_turn_q = None

    # ── Injection points (wired by main.py) ────────────────────────────────

    def register_tool_handler(self, fn: Callable[[str, dict], str]) -> None:
        self._tool_handler = fn

    def register_output_cb(self, fn: Callable[[str], None]) -> None:
        self._output_cb = fn

    def register_user_cb(self, fn: Callable[[str], None]) -> None:
        self._user_cb = fn

    def register_muted_getter(self, fn: Callable[[], bool]) -> None:
        self._muted_getter = fn

    def register_speaking_cb(self, fn: Callable[[bool], None]) -> None:
        self._speaking_cb = fn

    def set_system_prompt(self, prompt: str | None) -> None:
        self._system_instruction = prompt

    def set_tools(self, tools: list | None) -> None:
        if not tools:
            self._tools = None
            return
        from google.genai import types

        declarations = []
        for t in tools:
            func = t.get("function", t)
            declarations.append(
                types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters_json_schema=func.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    ),
                )
            )
        new_tools = [types.Tool(function_declarations=declarations)]

        # If tools changed and session is already connected, force a reconnect
        # so the new tools are included in the LiveConnectConfig.
        if self._tools != new_tools:
            self._tools = new_tools
            if self._session is not None and self._connected_event.is_set():
                print("[GeminiLive] Tools updated — reconnecting session…")
                self._close_current_session()
                self._connected_event.clear()

    def interrupt(self) -> None:
        """Stop mid-speech: drain queued audio and open the mic immediately."""
        self._interrupted = True
        q = self._audio_in_q
        if q is not None:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[GeminiLive] STOP Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        if self._turn_done_event is not None:
            self._turn_done_event.clear()

    def speak(self, text: str) -> None:
        """Send a text utterance into the live session (Mark-L's speak())."""
        if not self._loop or self._session is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_text(text), self._loop
        )

    @property
    def _uses_realtime_text(self) -> bool:
        """Use one realtime stream for current Live models and microphone audio."""
        return self._model.startswith("gemini-live-") or any(
            k in self._model for k in ("3.1", "3.2", "3.5", "gemini-3")
        )

    @property
    def _supports_compression(self) -> bool:
        """Enable long-session compression on current Live models.

        The old native-audio preview rejected this field, but its replacement
        (``gemini-live-2.5-flash``) supports it.
        """
        return self._model.startswith("gemini-live-") or any(
            k in self._model for k in ("3.1", "3.2", "3.5", "gemini-3")
        )

    async def _send_text(self, text: str) -> None:
        """Send a typed-text turn using send_client_content (Mark-L style)."""
        if self._uses_realtime_text:
            await self._session.send_realtime_input(text=text)
        else:
            await self._session.send_client_content(
                turns=[{"role": "user", "parts": [{"text": text}]}],
                turn_complete=True,
            )

    # ── Session thread ─────────────────────────────────────────────────────

    def _session_main(self) -> None:
        try:
            asyncio.run(self._session_loop())
        except Exception as e:
            print(f"[GeminiLive] Session thread ended: {e}")
            traceback.print_exc()
        finally:
            self._connected_event.clear()

    async def _session_loop(self) -> None:
        from google import genai

        backoff = self._conn_backoff
        while not self._stop.is_set():
            try:
                print("[GeminiLive] Connecting…")
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                client = genai.Client(
                    api_key=self._api_key,
                    http_options={"api_version": "v1beta"},
                )

                async with (
                    client.aio.live.connect(
                        model=f"models/{self._model}", config=config
                    ) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self._session          = session
                    self._audio_in_q       = asyncio.Queue()
                    self._out_q            = asyncio.Queue(maxsize=200)
                    self._turn_done_event  = asyncio.Event()
                    self._interrupted      = False
                    self._text_turn_active = False
                    self._tool_round         = False
                    self._content_after_tool = False
                    self._turn_acc         = ""
                    self._sent_buf         = ""
                    self._loop             = asyncio.get_event_loop()
                    self._connected_event.set()
                    print("[GeminiLive] Connected.")

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    print("[GeminiLive] All tasks started.")
            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                err_str = str(e)
                transient = _is_transient_connection_error(err_str)
                label = "Transient connection loss" if transient else "Error"
                print(f"[GeminiLive] {label} ({type(e).__name__}): {err_str}")
                if not transient:
                    traceback.print_exc()
                is_quota = _is_quota_error(err_str)
                # Record quota error for early detection in chat_stream
                if is_quota:
                    self._quota_error_message = err_str
                    self._stop.set()
                err_lower = err_str.lower()
                self._emit_error(
                    f"Gemini Live session error: {err_str}",
                    quota=is_quota,
                )
                if (
                    "not found" in err_lower
                    or "invalid session handle" in err_lower
                ):
                    backoff = self._conn_backoff
                elif transient:
                    backoff = min(backoff * 2, 60)
                else:
                    backoff = self._conn_backoff
            finally:
                self._session = None
                self._connected_event.clear()
                self.set_speaking(False)

            print(f"[GeminiLive] Reconnecting in {backoff}s…")
            await asyncio.sleep(backoff)

    def _build_config(self):
        from google.genai import types

        kwargs: dict = dict(
            response_modalities=[types.Modality.AUDIO],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=self._system_instruction,
            tools=self._tools,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice
                    )
                )
            ),
        )
        if self._supports_compression:
            kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
                trigger_tokens=25000,
                sliding_window=types.SlidingWindow(target_tokens=8000),
            )
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if self._resumption_handle:
            kwargs["session_resumption"] = types.SessionResumptionConfig(
                handle=self._resumption_handle
            )
        return types.LiveConnectConfig(**kwargs)

    # ── Tasks inside the connected session ─────────────────────────────────

    async def _send_realtime(self) -> None:
        while True:
            msg = await self._out_q.get()
            await self._session.send_realtime_input(
                media={"data": msg, "mime_type": "audio/pcm;rate=16000"}
            )

    async def _drain_realtime_input(self) -> None:
        """Remove queued microphone frames before a typed client-content turn."""
        if self._out_q is None:
            return
        while True:
            try:
                self._out_q.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _listen_audio(self) -> None:
        print("[GeminiLive] MIC Mic started")
        loop = asyncio.get_event_loop()

        def push(data: bytes) -> None:
            try:
                self._out_q.put_nowait(data)
            except asyncio.QueueFull:
                pass

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or self._text_turn_active:
                return
            try:
                if self._muted_getter is not None and self._muted_getter():
                    return
            except Exception:
                pass
            loop.call_soon_threadsafe(push, indata.tobytes())

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ) as stream:
                self._mic = stream
                print("[GeminiLive] MIC Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[GeminiLive] ERR Mic: {e}")
            raise
        finally:
            self._mic = None

    async def _receive_audio(self) -> None:
        print("[GeminiLive] RECV Recv started")
        out_buf: list[str] = []

        try:
            while True:
                async for response in self._session.receive():
                    # Current google-genai exposes native audio in
                    # server_content.model_turn.parts[*].inline_data, not on
                    # the top-level response. Accessing response.data caused
                    # every normal audio response to crash the receiver.
                    sc = response.server_content
                    model_turn = sc.model_turn if sc else None
                    parts = model_turn.parts if model_turn and model_turn.parts else []
                    audio_parts = [
                        part.inline_data.data for part in parts
                        if part.inline_data and part.inline_data.data
                    ]
                    for data in audio_parts:
                        if self._interrupted:
                            break
                        for i in range(0, len(data), 2400):
                            self._audio_in_q.put_nowait(data[i : i + 2400])

                    update = response.session_resumption_update
                    if update and update.resumable and update.new_handle:
                        self._resumption_handle = update.new_handle

                    if sc:
                        if sc.interrupted:
                            self._interrupted = True
                            self._drain_audio_output()
                            self.set_speaking(False)
                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)
                                q = self._active_turn_q
                                if q is not None:
                                    self._emit_sentences(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt and self._user_cb is not None:
                                try:
                                    self._user_cb(txt)
                                except Exception:
                                    pass

                        if sc.turn_complete:
                            if self._turn_done_event is not None:
                                self._turn_done_event.set()

                            if self._interrupted:
                                self._interrupted = False
                                out_buf = []
                                continue

                            full_out = " ".join(out_buf).strip()
                            out_buf = []
                            q = self._active_turn_q
                            if q is not None:
                                self._flush_sentence_buffer()
                                q.put({"type": "done", "content": full_out, "tool_calls": []})
                            elif full_out and self._output_cb is not None:
                                try:
                                    self._output_cb(full_out)
                                except Exception:
                                    pass

                    if response.tool_call:
                        self._tool_round = True
                        self._content_after_tool = False
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[GeminiLive] CALL {fc.name}")
                            result = await self._exec_tool(fc)
                            fn_responses.append(result)
                        await self._session.send_tool_response(
                            function_responses=fn_responses
                        )
                # The SDK ends one receive() iterator after a completed turn.
                # Stay in the outer loop and open the next iterator on the
                # same WebSocket; raising here used to disconnect after every
                # reply and forced an unnecessary full reconnect.
                continue
        except Exception as e:
            if _is_transient_connection_error(str(e)):
                print(f"[GeminiLive] Receiver connection lost; reconnecting: {e}")
            else:
                print(f"[GeminiLive] ERR Recv: {e}")
                traceback.print_exc()
            raise

    async def _exec_tool(self, fc):
        from google.genai import types

        name = fc.name
        args = dict(fc.args or {})
        if self._tool_handler is not None:
            try:
                result = await asyncio.to_thread(self._tool_handler, name, args)
            except Exception as e:
                traceback.print_exc()
                result = f"Tool '{name}' failed: {e}"
        else:
            result = f"Tool '{name}' is not available in this session."
        if isinstance(result, str) and result.strip() == "__SILENT__":
            return types.FunctionResponse(
                id=fc.id, name=name, response={"result": "ok", "silent": True}
            )
        if result is None:
            result = "Done."
        if not isinstance(result, str):
            result = str(result)
        if len(result) > 8000:
            result = result[:8000] + "\n…[truncated]"
        print(f"[GeminiLive] SENT {name} → {result[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name, response={"result": result}
        )

    async def _play_audio(self) -> None:
        print("[GeminiLive] PLAY Play started")
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        self._out_stream = stream
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_in_q.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event is not None
                        and self._turn_done_event.is_set()
                        and self._audio_in_q.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)

                # Batch immediately-available chunks into one write (≈200 ms cap
                # so interrupt() still stops audio within ~200 ms).
                batch = bytearray(chunk)
                while len(batch) < 9600:
                    try:
                        batch.extend(self._audio_in_q.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, OSError, sd.PortAudioError):
                    # Stream aborted externally (e.g. sd.stop()) — recreate it
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = sd.RawOutputStream(
                        samplerate=RECEIVE_SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="int16",
                        blocksize=CHUNK_SIZE,
                    )
                    self._out_stream = stream
                    stream.start()
                except asyncio.CancelledError:
                    break
        except Exception as e:
            print(f"[GeminiLive] ERR Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self._out_stream = None

    # ── Event bridge → active chat_stream turn ─────────────────────────────

    def _emit_sentences(self, text: str) -> None:
        self._sent_buf += text
        while True:
            m = _SENT_END.search(self._sent_buf)
            if not m:
                break
            sentence = self._sent_buf[: m.start() + 1].strip()
            self._sent_buf = self._sent_buf[m.end() :]
            if sentence:
                self._emit_event(
                    {"type": "sentence", "text": sentence, "native_audio": True}
                )

    def _flush_sentence_buffer(self) -> None:
        """Emit the final transcript fragment when it lacks punctuation."""
        sentence = self._sent_buf.strip()
        self._sent_buf = ""
        if sentence:
            self._emit_event(
                {"type": "sentence", "text": sentence, "native_audio": True}
            )

    def _drain_audio_output(self) -> None:
        q = self._audio_in_q
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _emit_done(self, full_out: str) -> None:
        q = self._active_turn_q
        acc = (self._turn_acc + " " + full_out).strip()
        self._turn_acc = ""
        if q is not None:
            q.put({"type": "done", "content": acc, "tool_calls": []})
            return
        # Voice-initiated turn (no chat_stream active) → deliver via callback
        if full_out and self._output_cb is not None:
            try:
                self._output_cb(full_out)
            except Exception:
                pass

    def _emit_error(self, message: str, quota: bool = False) -> None:
        if not quota:
            # Mark-L style: routine session errors are console-only — the
            # session loop already printed them, and reconnection happens on
            # its own. Never surface them to the app.
            return
        q = self._active_turn_q
        if q is not None:
            q.put({"type": "error", "message": message, "quota": quota})

    def _emit_event(self, ev: dict) -> None:
        q = self._active_turn_q
        if q is not None:
            q.put(ev)

    # ── Helpers ────────────────────────────────────────────────────────────

    def set_speaking(self, value: bool) -> None:
        with self._speaking_lock:
            self._is_speaking = value
        if self._speaking_cb is not None:
            try:
                self._speaking_cb(value)
            except Exception:
                pass

    def _build_live_input(self, messages: list) -> str:
        """Build a single user input that preserves conversation context."""
        parts = []
        for m in messages:
            role = m.get("role", "")
            if role == "system":
                continue
            text = m.get("content", "").strip()
            if text:
                parts.append(text)
        if not parts:
            return ""
        if len(parts) > 1:
            return "\n\n".join(parts)
        return parts[0]
