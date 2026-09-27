# ORTHOS

> Not just an AI, but a digital extension of the human will.

Personal AI desktop assistant with voice interaction, multi-LLM support, screen vision, and PyQt6 HUD.

---

## Features

- **Voice Interaction**: STT (Whisper/Groq/Gemini) + TTS (EdgeTTS/Kokoro/ElevenLabs/Gemini Live)
- **Multi-LLM Support**: Ollama, Groq, Gemini, Cloudflare, OpenRouter, NVIDIA
- **Screen Vision**: Moondream 3 / Qwen-VL for real-time screen understanding
- **OS Automation**: Playwright browser control, PyAutoGUI, file operations, terminal
- **Memory System**: ChromaDB vector search + SQLite conversations + knowledge graph
- **Telegram Gateway**: Remote bot interface
- **PyQt6 HUD**: Animated overlay with real-time system metrics
- **MCP Tools**: Extensible tool system with dynamic discovery

## Quick Start

```bash
git clone https://github.com/Ankur1103/Orthos.git
cd Orthos
pip install -r requirements.txt
python main.py
```

Or install as a package:

```bash
pip install -e ".[dev]"
```

## Project Structure

```
orthos/
├── main.py              # Entry point
├── ui.py                # PyQt6 GUI (HUD, chat, setup)
├── core/                # Core library (LLM, TTS, STT, security, observability)
├── actions/             # LLM-callable tool actions
├── memory/              # Memory system (ChromaDB, SQLite, knowledge graph)
├── gateway/             # Multi-platform messaging gateway
├── agent/               # Agent subsystem (planner, executor, task queue)
├── config/              # Configuration files
├── tests/               # Test suite
├── assets/              # Static assets
└── models/              # Model artifacts
```

## Configuration

1. Copy `.env.example` to `.env`
2. Fill in your API keys
3. Or edit `config/api_keys.json` directly

See `.env.example` for all available options.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Architected by Ankur & Orthos**
