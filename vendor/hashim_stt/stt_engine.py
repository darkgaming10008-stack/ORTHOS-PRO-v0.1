import os
import sys
import time
import queue
import shutil
import asyncio
import threading
import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MODEL = "models/gemini-3.5-live-translate-preview"


class STT:
    def __init__(self, target_lang: str = None, flush: bool = False, silence_threshold: float = 1.2, api_key: str = None):
        self.target_lang = target_lang
        self.flush = flush
        self.silence_threshold = silence_threshold
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        self.session = None
        self._cm = None
        self.pya = None
        self.audio_stream = None

        self.last_text_time = 0
        self.current_sentence = ""
        self.sentence_queue = queue.Queue()
        self._is_running = False
        self._loop = None
        self._thread = None
        self.last_error = None
        self._gate = None

    def connect(self):
        if self._is_running:
            return
        self._is_running = True
        self.last_error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def listen(self) -> str:
        return self.sentence_queue.get()

    def is_connected(self) -> bool:
        """True while the engine thread is running normally."""
        return bool(self._thread and self._thread.is_alive())

    def set_gate(self, fn):
        """Optional speaking-gate (Mark-style): when fn() is True, mic frames
        are dropped instead of streamed — the host is talking (TTS) or the
        room tail is still ringing. Prevents self-echo transcription."""
        self._gate = fn

    def stop(self):
        """Fully stop: mic stream, live session, asyncio loop and thread.

        Only closes the PyAudio *stream* here (the PyAudio instance and
        event loop belong to the engine thread); the thread's finally-block
        tears the session down and terminates the loop. ``close()`` keeps
        its historical half-close semantics for compatibility.
        """
        self._is_running = False
        stream = self.audio_stream
        if stream:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass

    def close(self):
        if self.audio_stream:
            try:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
            except Exception:
                pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_connect())
        except Exception as e:
            # surface the REAL failure to callers instead of a bare
            # "thread died" (config errors, bad key, network, ...)
            self.last_error = e
            raise
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            self._is_running = False

    async def _async_connect(self):
        self.pya = pyaudio.PyAudio()

        # Build LiveConnectConfig via kwargs: optional fields must NOT be
        # passed as explicit None — pydantic v2 (extra_forbidden) rejects
        # unknown keys like translation_config when the installed
        # google-genai version doesn't declare them.
        cfg_kwargs = dict(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
        )
        if self.target_lang:
            cfg_kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
            cfg_kwargs["translation_config"] = types.TranslationConfig(
                target_language_code=self.target_lang,
                echo_target_language=True,
            )

        config = types.LiveConnectConfig(**cfg_kwargs)

        client_kwargs = {"http_options": {"api_version": "v1beta"}}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key

        client = genai.Client(**client_kwargs)

        self._cm = client.aio.live.connect(model=MODEL, config=config)
        self.session = await self._cm.__aenter__()

        tasks = [
            asyncio.create_task(self._listen_audio()),
            asyncio.create_task(self._receive_text()),
            asyncio.create_task(self._check_silence()),
        ]

        try:
            while self._is_running:
                await asyncio.sleep(0.1)
        finally:
            self._is_running = False
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._cm:
                try:
                    await self._cm.__aexit__(None, None, None)
                except Exception:
                    pass
            self.session = None
            self._cm = None
            stream = self.audio_stream
            self.audio_stream = None
            if stream:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            if self.pya:
                try:
                    self.pya.terminate()
                except Exception:
                    pass
                self.pya = None

    def _render_live_preview(self, text: str):
        if self.flush and text:
            cols = max(20, shutil.get_terminal_size().columns - 10)
            disp = ("..." + text[-cols:]) if len(text) > cols else text
            sys.stdout.write(f"\r\033[KLive: {disp}")
            sys.stdout.flush()

    async def _listen_audio(self):
        try:
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )

            while self._is_running:
                data = await asyncio.to_thread(
                    self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                )
                if self.session and self._is_running:
                    try:
                        gated = self._gate() if self._gate else False
                    except Exception:
                        gated = False
                    if gated:
                        continue          # host speaking / tail — drop frame
                    await self.session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm")
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._is_running:
                self.last_error = e

    async def _receive_text(self):
        try:
            async for response in self.session.receive():
                if not self._is_running or not response.server_content:
                    continue

                sc = response.server_content
                fragment = (
                    sc.output_transcription.text if self.target_lang and sc.output_transcription
                    else sc.input_transcription.text if not self.target_lang and sc.input_transcription
                    else None
                )

                if fragment:
                    self.current_sentence += fragment
                    self.last_text_time = time.monotonic()
                    self._render_live_preview(self.current_sentence)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._is_running:
                self.last_error = e

    async def _check_silence(self):
        try:
            while self._is_running:
                await asyncio.sleep(0.1)
                if self.current_sentence and (time.monotonic() - self.last_text_time > self.silence_threshold):
                    line = self.current_sentence.strip()
                    self.current_sentence = ""
                    if line:
                        if self.flush:
                            sys.stdout.write("\r\033[K")
                            sys.stdout.flush()
                        self.sentence_queue.put(line)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._is_running:
                self.last_error = e
