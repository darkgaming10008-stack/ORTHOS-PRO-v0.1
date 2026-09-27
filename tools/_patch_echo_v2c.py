"""One-shot patch part C: capture envelope = the transcribed utterance's own audio."""


def crlf(s):
    return s.replace('\n', '\r\n')


p = 'main.py'
src = open(p, encoding='utf-8', newline='').read()
n0 = len(src)


def sub(old, new, tag, allow_multiple=False):
    global src
    o, n = crlf(old), crlf(new)
    c = src.count(o)
    if allow_multiple:
        assert c >= 1, f'{tag}: count={c}'
    else:
        assert c == 1, f'{tag}: count={c}'
    src = src.replace(o, n)


# ── 1. callbacks: DROP the live capture push (wrong window — bleed during
#      playback is always high on TRRS and would let word-overlap reject
#      alone). The capture envelope comes from the utterance audio itself. ──
sub("""        def callback(indata, frames, time_info, status):
            # Echo-risk measurement: record how loud the room (and the
            # speakers) are in this mic while the TTS is playing.
            with self._speaking_lock:
                _spk = self._speaking
            if _spk and getattr(self, "_capture_env", None) is not None:
                try:
                    self._envelope_push(self._capture_env,
                                        indata.astype(np.float32) / 32768.0,
                                        SAMPLE_RATE_IN, self._ENV_RATE)
                except Exception:
                    pass
            if self._measuring_bleed:
""",
    """        def callback(indata, frames, time_info, status):
            # Echo-risk measurement: record how loud the room (and the
            # speakers) are in this mic while the TTS is playing.
""", 'cb_unwire', allow_multiple=True)

# ── 2. whisper main path: envelope from the flushed utterance ─────────────
sub("""                            audio = vad.process(chunk)
                            if audio is not None:
                                self.stop_speaking()
                                self._signal_cancel()
                                vad.reset()   # buffer is echo, drop it
                                try:
                                    text = self._stt.transcribe(audio)
""",
    """                            audio = vad.process(chunk)
                            if audio is not None:
                                self.stop_speaking()
                                self._signal_cancel()
                                vad.reset()   # buffer is echo, drop it
                                # The utterance's own loudness envelope is
                                # the capture side of the echo fingerprint.
                                self._capture_env = []
                                try:
                                    self._envelope_push(
                                        self._capture_env, audio,
                                        SAMPLE_RATE_IN, self._ENV_RATE)
                                except Exception:
                                    self._capture_env = []
                                try:
                                    text = self._stt.transcribe(audio)
""", 'whisper_main_env')

# ── 3. whisper barge path: same ────────────────────────────────────────────
sub("""                                        audio = vad.process(chunk)
                                        if audio is not None:
                                            try:
                                                text = self._stt.transcribe(audio)
""",
    """                                        audio = vad.process(chunk)
                                        if audio is not None:
                                            self._capture_env = []
                                            try:
                                                self._envelope_push(
                                                    self._capture_env, audio,
                                                    SAMPLE_RATE_IN, self._ENV_RATE)
                                            except Exception:
                                                self._capture_env = []
                                            try:
                                                text = self._stt.transcribe(audio)
""", 'whisper_barge_env')

# ── 4. vosk barge path: envelope from vad_audio ────────────────────────────
sub("""                                        vad_audio = vad.process(chunk)
                                        if vad_audio is not None:
                                            try:
                                                text, _ = self._stt.process_chunk(vad_audio.tobytes())
""",
    """                                        vad_audio = vad.process(chunk)
                                        if vad_audio is not None:
                                            self._capture_env = []
                                            try:
                                                self._envelope_push(
                                                    self._capture_env, vad_audio,
                                                    SAMPLE_RATE_IN, self._ENV_RATE)
                                            except Exception:
                                                self._capture_env = []
                                            try:
                                                text, _ = self._stt.process_chunk(vad_audio.tobytes())
""", 'vosk_barge_env')

# ── 5. vosk main path: accumulate the utterance's speech segments ─────────
sub("""                            vad_audio = vad.process(chunk)
                            if vad_audio is not None:
                                text, _ = self._stt.process_chunk(vad_audio.tobytes())
                                if text.strip():
                                    final_texts.append(text)
""",
    """                            vad_audio = vad.process(chunk)
                            if vad_audio is not None:
                                try:
                                    self._envelope_push(
                                        self._capture_env, vad_audio,
                                        SAMPLE_RATE_IN, self._ENV_RATE)
                                except Exception:
                                    self._capture_env = []
                                text, _ = self._stt.process_chunk(vad_audio.tobytes())
                                if text.strip():
                                    final_texts.append(text)
""", 'vosk_main_env')

open(p, 'wb').write(src.encode('utf-8'))
print(f'part C applied: {n0} -> {len(src)} chars')
