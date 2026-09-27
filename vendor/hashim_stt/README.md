# vendor/hashim_stt — vendored code (locally patched)

This directory contains a copy of:

**gemini-realtime-speech-to-text-translator** by Hashim Malik
https://github.com/hashimmalikdev/gemini-realtime-speech-to-text-translator
(MIT License — see LICENSE_SOURCE)

## Files

- `stt_engine.py` — copy of the repo's `stt_engine.py`
  (md5 9c4d67731b9fa36c22206502660d8f69 at vendoring time), **with four
  minimal local fixes (2026-09, md5 cc8ed0bc2bc1d330165affbaf3003436):**
  1. `LiveConnectConfig` is built without `None` extras — pydantic 2.13
     rejects `translation_config=None` / `output_audio_transcription=None`
     (`extra_forbidden`), which killed the engine thread at connect().
  2. The Gemini client takes the API key explicitly (`STT(api_key=…)`),
     falling back to `GEMINI_API_KEY` as before.
  3. `stop()` fully tears the session down (mic stream + live session +
     asyncio loop) and `last_error` records the exception that killed the
     thread, so Orthos can hot-swap STT engines and show the real cause
     instead of "thread died".
  4. `set_gate(fn)` — Mark-style speaking-gate: while `fn()` is True
     (host TTS playing / room tail), mic frames are dropped instead of
     streamed, so the live session never hears Orthos's own voice.

  If you re-vendor a new upstream version, re-apply these fixes and bump
  `VENDOR_MD5` in `tests/test_hashim_stt.py`.

## How Orthos uses it

`core/stt.py::HashimSTT` is a thin adapter that drives the vendored `STT`
class: `.connect()` starts the engine's own background thread, mic
stream and Gemini Live session; sentences are pumped from the engine's
`sentence_queue`. The engine choice in Orthos is `stt_engine = "hashim"`
(shown as "Hashim" in the UI).

Why it exists here: this engine streams mic audio to Gemini Live, so
voice-activity detection and echo suppression happen server-side — the
same property the Orthos project relies on.
