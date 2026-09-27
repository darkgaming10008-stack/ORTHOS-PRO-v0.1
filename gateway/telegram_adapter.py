"""
Orthos — Telegram Adapter

Uses python-telegram-bot v20.x (Application class) to:
  • Receive user messages → route to MessageRouter
  • Send assistant responses back to the same chat via ResponseRouter
  • Backend mode: capture terminal output and forward to a log channel

Runs on the shared AsyncWorker event loop.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from .async_worker import get_async_worker

_BOT_STATE_PATH = Path(__file__).resolve().parent.parent / "config" / ".bot_state.json"
_DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "config" / "telegram_downloads"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".heic"}
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # Telegram Bot API getFile limit


class TelegramAdapter:
    """
    Platform adapter for Telegram.

    Args:
        token:         Bot token from BotFather.
        on_message:    Callback(text, source="telegram", chat_id=int, ...).
                       Not required in backend_mode.
        log:           Optional logging function.
        backend_mode:  If True, this adapter only forwards terminal logs
                       to the chat that sent /start.  Ignores other messages.
    """

    def __init__(
        self,
        token: str,
        on_message: Callable | None = None,
        log: Callable[[str], None] | None = None,
        backend_mode: bool = False,
    ):
        self._token = token
        self._on_message = on_message
        self._log = log or print
        self._backend_mode = backend_mode
        self._backend_chat_id: int | None = None
        self._app: Application | None = None
        self._running = False
        self._last_chat_id: int | None = None

        self._load_backend_chat_id()

    def _load_backend_chat_id(self):
        try:
            if _BOT_STATE_PATH.exists():
                state = json.loads(_BOT_STATE_PATH.read_text(encoding="utf-8"))
                self._backend_chat_id = state.get("backend_chat_id")
        except Exception:
            pass

    def _save_backend_chat_id(self, chat_id: int):
        try:
            _BOT_STATE_PATH.write_text(
                json.dumps({"backend_chat_id": chat_id}),
                encoding="utf-8"
            )
        except Exception:
            pass

    def start(self):
        """Launch the Telegram bot on the shared AsyncWorker loop."""
        if self._running:
            return
        self._running = True

        self._app = Application.builder().token(self._token).build()

        self._app.add_handler(CommandHandler("start", self._cmd_start))

        if not self._backend_mode:
            self._app.add_handler(CommandHandler("help", self._cmd_help))
            self._app.add_handler(MessageHandler(filters.ALL, self._handle_message))
            self._log("TELEGRAM: Starting polling…")
        else:
            self._log("TELEGRAM (backend): Starting polling…")

        worker = get_async_worker()
        worker.run(self._start_async())

    async def _start_async(self):
        """Initialize the Application and begin polling on the shared loop."""
        try:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            self._log(f"TELEGRAM: Start error — {e}")
            self._app = None
            self._running = False

    def stop(self):
        """Shut down the bot gracefully."""
        self._running = False
        if not self._app:
            return
        worker = get_async_worker()
        try:
            worker.run_sync(self._stop_async(), timeout=10)
        except Exception as e:
            self._log(f"TELEGRAM: Shutdown error — {e}")

    async def _stop_async(self):
        if not self._app:
            return
        try:
            if self._app.updater:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        finally:
            self._app = None

    async def _cmd_start(self, update: Update, _context):
        if self._backend_mode:
            self._backend_chat_id = update.effective_chat.id
            self._save_backend_chat_id(self._backend_chat_id)
            self._log(f"LOG CHANNEL: Backend bot registered chat_id={self._backend_chat_id}")
            await update.message.reply_text(
                "LOG CHANNEL ACTIVE\n\n"
                "All terminal output from Orthos will be forwarded here.\n"
                "This bot does not process commands or messages."
            )
            return
        user = update.effective_user
        name = user.first_name if user else "there"
        await update.message.reply_text(
            f"Hi {name}! I'm Orthos.\n"
            "Send me a message and I'll route it to my AI brain.\n"
            "Use /help to see what I can do."
        )

    async def _cmd_help(self, update: Update, _context):
        await update.message.reply_text(
            "I'm a multi-platform AI assistant.\n"
            "Just type anything — I'll process it with my LLM and tools.\n\n"
            "Available commands:\n"
            "/start — Greeting\n"
            "/help  — This message\n\n"
            "Tip: Send me images, PDFs, videos, voice notes, GIFs, stickers,\n"
            "polls or locations — I'll handle them."
        )

    async def _handle_message(self, update: Update, context):
        """Catch-all dispatcher: text, media, and non-file content."""
        if not update.message:
            return
        msg = update.message

        has_attachment = any([
            msg.photo, msg.document, msg.video, msg.video_note,
            msg.animation, msg.audio, msg.voice, msg.sticker,
        ])
        has_content = any([
            has_attachment, msg.text, msg.location, msg.venue,
            msg.contact, msg.poll, msg.dice,
        ])
        if not has_content:
            return  # service / status update — ignore silently

        if has_attachment:
            await self._handle_media(update, context)
        elif msg.text:
            await self._handle_text(update, context)
        else:
            await self._handle_non_file(update, context)

    async def _handle_text(self, update: Update, _context):
        if not update.message or not update.message.text:
            return
        text = update.message.text.strip()
        if not text:
            return

        chat_id = update.effective_chat.id if update.effective_chat else None
        self._last_chat_id = chat_id

        self._log(f"TELEGRAM <{chat_id}>: {text[:60].encode('ascii', 'replace').decode()}")

        self._on_message(
            text,
            source="telegram",
            chat_id=chat_id,
            user_id=update.effective_user.id if update.effective_user else None,
        )

    async def _handle_non_file(self, update: Update, _context):
        """Location, venue, contact, poll, dice → describe as text so Orthos sees it."""
        msg = update.message
        chat_id = update.effective_chat.id if update.effective_chat else None
        self._last_chat_id = chat_id
        user_id = update.effective_user.id if update.effective_user else None

        if msg.location:
            text = f"📍 User shared a location: {msg.location.latitude}, {msg.location.longitude}"
            tag = "location"
        elif msg.venue:
            text = (f"📍 User shared a venue: {msg.venue.title} — {msg.venue.address}\n"
                    f"Coordinates: {msg.venue.location.latitude}, {msg.venue.location.longitude}")
            tag = "venue"
        elif msg.contact:
            name = f"{msg.contact.first_name} {msg.contact.last_name or ''}".strip()
            text = f"👤 User shared a contact: {name} — {msg.contact.phone_number or 'no phone'}"
            tag = "contact"
        elif msg.poll:
            options = ", ".join(o.text for o in (msg.poll.options or [])[:10])
            text = f"📊 User sent a poll: '{msg.poll.question}' — options: {options}"
            tag = "poll"
        elif msg.dice:
            text = f"🎲 User rolled a {msg.dice.emoji}: {msg.dice.value}"
            tag = "dice"
        else:
            return

        self._log(f"TELEGRAM <{chat_id}>: [{tag}] {text[:60].encode('ascii', 'replace').decode()}")
        self._on_message(text, source="telegram", chat_id=chat_id, user_id=user_id)

    async def _handle_media(self, update: Update, context):
        """Receive any attachment, download it, and route to the AI."""
        if not update.message:
            return
        msg = update.message

        file = None
        file_name = None
        is_image = False
        media_type = "file"

        if msg.photo:
            file = msg.photo[-1]
            file_name = f"{file.file_unique_id}.jpg"
            is_image = True
            media_type = "photo"
        elif msg.sticker:
            st = msg.sticker
            media_type = "sticker"
            if st.is_animated:
                file, file_name = st, f"{st.file_unique_id}.tgs"
            elif st.is_video:
                file, file_name = st, f"{st.file_unique_id}.webm"
            else:
                file, file_name = st, f"{st.file_unique_id}.webp"
                is_image = True
        elif msg.document:
            doc = msg.document
            ext = Path(doc.file_name or "").suffix.lower()
            mime = (doc.mime_type or "").lower()
            is_image = ext in _IMAGE_EXTS or mime.startswith("image/")
            file = doc
            file_name = doc.file_name or f"{doc.file_unique_id}{ext or '.bin'}"
            media_type = "document"
        elif msg.video:
            file = msg.video
            file_name = f"{file.file_unique_id}.mp4"
            media_type = "video"
        elif msg.video_note:
            file = msg.video_note
            file_name = f"{file.file_unique_id}.mp4"
            media_type = "video_note"
        elif msg.animation:
            file = msg.animation
            file_name = f"{file.file_unique_id}.gif"
            media_type = "animation"
        elif msg.audio:
            file = msg.audio
            file_name = msg.audio.file_name or f"{file.file_unique_id}.mp3"
            media_type = "audio"
        elif msg.voice:
            file = msg.voice
            file_name = f"{file.file_unique_id}.ogg"
            media_type = "voice"

        if file is None:
            return

        chat_id = update.effective_chat.id if update.effective_chat else None
        self._last_chat_id = chat_id
        user_id = update.effective_user.id if update.effective_user else None

        size = getattr(file, "file_size", 0) or 0
        if size > _MAX_DOWNLOAD_BYTES:
            self._log(f"TELEGRAM <{chat_id}>: File too large ({size / 1e6:.1f} MB) — {file_name}")
            if chat_id:
                self.send_message(
                    chat_id,
                    "⚠️ That file is over Telegram's 20 MB bot-download limit, "
                    "so I can't fetch it. Send a smaller version.",
                )
            return

        try:
            _DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = _DOWNLOAD_DIR / f"{stamp}_{Path(file_name).name}"
            tg_file = await context.bot.get_file(file.file_id)
            await tg_file.download_to_drive(str(path))
        except Exception as e:
            self._log(f"TELEGRAM: Media download error — {e}")
            if chat_id:
                self.send_message(chat_id, "⚠️ Sorry, I couldn't download that. Please try again.")
            return

        if media_type == "voice":
            await update.message.reply_text("🎤 Transcribing your voice note…")
            threading.Thread(
                target=self._transcribe_voice_thread,
                args=(str(path), chat_id, user_id),
                daemon=True,
            ).start()
            return

        if is_image:
            text = (msg.caption or "Image received — please analyze this image.").strip()
        else:
            text = (msg.caption or "File received — process this file if needed.").strip()
        self._log(f"TELEGRAM <{chat_id}>: [{media_type}] {path.name}")
        extra = {
            "chat_id": chat_id,
            "user_id": user_id,
            "files": [str(path)],
            "media_type": media_type,
        }
        if is_image:
            extra["image_files"] = [str(path)]
        self._on_message(text, source="telegram", **extra)

    def _transcribe_voice_thread(self, path: str, chat_id: int, user_id: int | None):
        """Decode the .ogg voice note and transcribe it with the configured STT engine."""
        try:
            stt = self._load_stt()
            if stt is None:
                raise RuntimeError("No STT engine configured")
            import soundfile as sf
            try:
                audio, sr = sf.read(path, dtype="float32")
            except Exception:
                out = Path(tempfile.mktemp(suffix=".wav"))
                subprocess.run(
                    ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1",
                     "-c:a", "pcm_s16le", str(out)],
                    capture_output=True, timeout=180, check=True,
                )
                if not out.exists():
                    raise RuntimeError("Could not decode voice note (ffmpeg needed)")
                audio, sr = sf.read(str(out), dtype="float32")
                out.unlink(missing_ok=True)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            transcript = stt.transcribe(audio).strip()
            if transcript:
                self._log(f"TELEGRAM <{chat_id}>: [voice transcript] {transcript[:60]}")
                self._on_message(
                    f"🎤 Voice note transcript: {transcript}",
                    source="telegram",
                    chat_id=chat_id,
                    user_id=user_id,
                    files=[path],
                    media_type="voice",
                )
                return
        except Exception as e:
            self._log(f"TELEGRAM: Voice transcription failed — {e}")
        self._on_message(
            "Voice note received — transcribe or analyze it if needed.",
            source="telegram",
            chat_id=chat_id,
            user_id=user_id,
            files=[path],
            media_type="voice",
        )

    def _load_stt(self):
        """Build the STT engine from config/api_keys.json (same as runtime)."""
        try:
            cfg_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            engine = (cfg.get("stt_engine") or "").lower()
            language = cfg.get("stt_language", "auto")
            if engine in ("groq",):
                key = cfg.get("groq_api_key") or ""
                if key:
                    from core.stt import GroqSTT
                    return GroqSTT(key, language)
            if engine in ("whisper", "faster-whisper"):
                from core.stt import WhisperSTT
                return WhisperSTT(cfg.get("stt_model", "base"), language)
        except Exception as e:
            self._log(f"TELEGRAM: STT load failed — {e}")
        return None

    def send_response(self, text: str, source: str) -> None:
        """
        Called by ResponseRouter when a response is ready.
        Only sends if *source* is "telegram" and we have a known chat.
        Sends images referenced as [SEND_IMAGE: /path/to/file] in the text.
        """
        if source != "telegram" or self._last_chat_id is None:
            return
        images = [p.strip() for p in re.findall(r"\[SEND_IMAGE:\s*([^\]]+)\]", text)]
        clean = re.sub(r"\[SEND_IMAGE:\s*[^\]]+\]", "", text).strip()
        if images:
            self.send_photo(self._last_chat_id, images[0], caption=clean)
            for extra_img in images[1:4]:
                self.send_photo(self._last_chat_id, extra_img)
        else:
            self.send_message(self._last_chat_id, text)

    def send_log(self, text: str) -> None:
        """Forward a terminal log line to the backend log channel."""
        if self._backend_chat_id is None:
            return
        if not self._app:
            return
        MAX_LEN = 4096
        payload = text[:MAX_LEN] if len(text) > MAX_LEN else text
        worker = get_async_worker()

        async def _send():
            try:
                await self._app.bot.send_message(
                    chat_id=self._backend_chat_id,
                    text=payload,
                )
            except Exception as e:
                self._log(f"TELEGRAM: Send log error — {e}")

        worker.run(_send())

    def send_message(self, chat_id: int, text: str) -> None:
        """Send a text response to a Telegram chat."""
        if not self._app:
            return
        worker = get_async_worker()

        async def _send():
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                )
            except Exception as e:
                self._log(f"TELEGRAM: Send error — {e}")

        worker.run(_send())

    def send_photo(self, chat_id: int, photo_path: str, caption: str = "") -> None:
        """Send an image file to a Telegram chat."""
        if not self._app:
            return
        path = Path(photo_path)
        if not path.is_file():
            self._log(f"TELEGRAM: Image not found — {photo_path}")
            return
        worker = get_async_worker()

        async def _send():
            try:
                with path.open("rb") as fh:
                    await self._app.bot.send_photo(
                        chat_id=chat_id,
                        photo=fh,
                        caption=(caption or None),
                    )
            except Exception as e:
                self._log(f"TELEGRAM: Send photo error — {e}")

        worker.run(_send())
