@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Orthos - One-Click Setup

REM ============================================================================
REM   ORTHOS  -  ONE-CLICK SETUP
REM ----------------------------------------------------------------------------
REM   Non-tech user ke liye: bas is file ko DOUBLE-CLICK karo. Baaki sab ye khud
REM   karega, chahe PC par kuchh bhi install na ho. KUCHH BHI NAHI CHHUTEGA:
REM
REM   [1]  Python 3.12 dhoondhega - na mile to khud download + install karega
REM        (per-user install, koi admin password nahi maangega)
REM   [2]  Project ke andar .venv banayega (PC ka Python gandaa nahi hoga)
REM   [3]  requirements.txt ke saare packages install karega
REM   [4]  Playwright Chromium browser (browser-control feature ke liye)
REM   [5]  .env file (.env.example se) + config/api_keys.json default settings
REM   [6]  Runtime folders + MEMORY SYSTEM ka poora init:
REM        - conversations.db (SQLite) init  - chroma_db (vector store) init
REM        - Memory Service (FastAPI, port 9876) ka LIVE smoke test
REM   [7]  TTS/STT verify - faster-whisper, edge-tts, gTTS, Groq STT, aur
REM        HASHIM TTS (google-genai + pyaudio + python-dotenv) - jo missing
REM        ho wo yahin install ho jayega
REM   [8]  Models PRE-DOWNLOAD (ek baar): Whisper 'base' STT (~150 MB),
REM        MiniLM embeddings (~90 MB), Silero VAD (2.3 MB) - isse pehli
REM        launch 5-15 minute ki jagah ~1 minute me hoti hai
REM   [9]  EXTRA optional tools: Node.js (npx) + uv (uvx) MCP ke liye,
REM        Git, ffmpeg, Ollama  -> inke bina bhi Orthos chalta hai
REM   [10] Desktop shortcut (optional) aur Orthos start (optional)
REM
REM   Safe hai dobara chalane par bhi - jo pehle se hai wo skip ho jata hai.
REM   Poora reinstall chahiye:        setup.bat /reinstall
REM   Sirf test (kuchh install nahi): setup.bat /dryrun
REM   Sirf extra tools install:       setup.bat /extras
REM ============================================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "LOG=%ROOT%setup_log.txt"
set "TMPPY=%TEMP%\orthos_pyver.txt"
set "TMPDL=%TEMP%\orthos_python_setup.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYEXE="
set "VPY="
set "PYVER="
set "WARNED="
set "DRYRUN=0"
set "REINSTALL=0"
set "EXTRAS="
REM Saare arguments check hote hain, isliye "setup.bat /dryrun /extras" bhi chalta hai.
for %%A in (%*) do (
    if /i "%%~A"=="/reinstall"  set "REINSTALL=1"
    if /i "%%~A"=="-reinstall"  set "REINSTALL=1"
    if /i "%%~A"=="--reinstall" set "REINSTALL=1"
    if /i "%%~A"=="/dryrun"    set "DRYRUN=1"
    if /i "%%~A"=="-dryrun"    set "DRYRUN=1"
    if /i "%%~A"=="/extras"    set "EXTRAS=1"
)
if /i "%~1"=="/help" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage

cls
echo.
echo  ==============================================================
echo    ORTHOS  -  ONE-CLICK SETUP
echo    Sab kuchh automatic - koi technical knowledge nahi chahiye
echo  ==============================================================
echo.
echo   Ye setup ye sab karega:
echo     - Python 3.12  (na ho to khud install karega - admin ki zarurat nahi)
echo     - Orthos ke saare packages (UI, voice, memory, browser, TTS/STT)
echo     - Chromium browser engine (Playwright)
echo     - Memory system ka poora setup + live test (SQLite + ChromaDB + FastAPI)
echo     - Hashim TTS + Whisper STT + EdgeTTS - sab voice engines ready
echo     - Models pehle se download (Whisper + MiniLM + VAD) - pehli launch fast
echo     - Config files (.env + config/api_keys.json) aur runtime folders
echo     - Extra tools (optional: Node.js, uv, Git, ffmpeg, Ollama)
echo.
echo   Samay: ~15 se 45 minute (internet speed par depend karta hai).
echo   Itne samay tak window BAND MAT karna. Progress: setup_log.txt
echo.

if not exist "%ROOT%main.py" goto :err_wrong_folder
if not exist "%ROOT%requirements.txt" goto :err_wrong_folder
echo [%date% %time%] Orthos setup started >"%LOG%"
if "%EXTRAS%"=="1" goto :extras_stage

REM ============================================================================
REM  [1/10]  Python dhoondho
REM ============================================================================
echo  [1/10] Python dhoondh raha hoon...

call :probe py -3.12
if defined PYEXE goto :python_found
call :probe py -3.11
if defined PYEXE goto :python_found
call :probe py -3.13
if defined PYEXE goto :python_found
call :probe py
if defined PYEXE goto :python_found
call :probe python
if defined PYEXE goto :python_found
call :probe "%LocalAppData%\Programs\Python\Python312\python.exe"
if defined PYEXE goto :python_found
call :probe "%LocalAppData%\Programs\Python\Python311\python.exe"
if defined PYEXE goto :python_found
call :probe "%LocalAppData%\Programs\Python\Python313\python.exe"
if defined PYEXE goto :python_found
call :probe "C:\Program Files\Python312\python.exe"
if defined PYEXE goto :python_found
call :probe "C:\Python312\python.exe"
if defined PYEXE goto :python_found
goto :install_python

:python_found
"%PYEXE%" -c "import sys;print('.'.join(str(n) for n in sys.version_info[:3]))" >"%TMPPY%" 2>nul
set /p PYVER=<"%TMPPY%"
echo        [OK] Python %PYVER% mil gaya
echo        Path: %PYEXE%
goto :venv_stage

REM ---------------------------------------------------------------------------
REM  Python nahi mila -> khud install karo (per-user, koi admin prompt nahi)
REM ---------------------------------------------------------------------------
:install_python
echo        [INFO] PC par Python nahi mila - Orthos khud install kar dega.
echo.
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan Python install hota.& set "PYEXE=C:\fake\python.exe"& goto :venv_stage)

REM --- Koshish 1: winget (Windows 10/11 par built-in hota hai) ---------------
where winget >nul 2>&1
if errorlevel 1 goto :py_download
echo        [1a] winget se Python 3.12 install kar raha hoon...
winget install --id Python.Python.3.12 --exact --scope user --silent --accept-package-agreements --accept-source-agreements >>"%LOG%" 2>&1
call :refresh_path
call :probe py -3.12
if defined PYEXE goto :py_installed
call :probe "%LocalAppData%\Programs\Python\Python312\python.exe"
if defined PYEXE goto :py_installed
call :probe python
if defined PYEXE goto :py_installed
echo        [WARN] winget se nahi hua - direct download try kar raha hoon...

:py_download
REM --- Koshish 2: python.org se official installer ---------------------------
echo        [1b] python.org se Python 3.12 download kar raha hoon (~27 MB)...
set "PYURL="
for %%V in (3.12.10 3.12.9 3.12.8 3.12.7) do (
    if not defined PYURL (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/%%V/python-%%V-amd64.exe' -OutFile $env:TEMP+'\orthos_python_setup.exe' -UseBasicParsing } catch { exit 1 }" >>"%LOG%" 2>&1
        if not errorlevel 1 if exist "%TMPDL%" set "PYURL=%%V"
    )
)
if not defined PYURL goto :py_failed

echo        Download complete - silent install chal raha hai (2-4 minute)...
start /wait "" "%TMPDL%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 InstallLauncherAllUsers=0 Include_test=0 Include_doc=0 Include_tcltk=1 SimpleInstall=1
del "%TMPDL%" >nul 2>&1
call :refresh_path
call :probe py -3.12
if defined PYEXE goto :py_installed
call :probe "%LocalAppData%\Programs\Python\Python312\python.exe"
if defined PYEXE goto :py_installed
call :probe python
if defined PYEXE goto :py_installed

:py_failed
echo.
echo  ==============================================================
echo    Python apne aap install nahi ho paya
echo  ==============================================================
echo.
echo   Internet ya antivirus ki wajah se rok laga ho sakta hai. Aap ye karo:
echo.
echo   1. Microsoft Store khul jayega - wahan se "Python 3.12" install karo
echo   2. Uske baad isi setup.bat ko DOBARA double-click karo
echo.
echo   (python.org se bhi install kar sakte ho - installer me
echo    "Add python.exe to PATH" ka tick lagana mat bhoolna)
echo.
choice /C YN /T 20 /D Y /M "   Ab Microsoft Store khol doon? [Y/N]"
if errorlevel 2 goto :err_exit
start "" "ms-windows-store://search/?query=Python%%203.12"
goto :err_exit

:py_installed
"%PYEXE%" -c "import sys;print('.'.join(str(n) for n in sys.version_info[:3]))" >"%TMPPY%" 2>nul
set /p PYVER=<"%TMPPY%"
echo        [OK] Python %PYVER% install ho gaya
echo        Path: %PYEXE%

REM ============================================================================
REM  [2/10]  Project ka apna environment (.venv)
REM ============================================================================
:venv_stage
echo.
echo  [2/10] Orthos ka apna Python environment (.venv) taiyaar kar raha hoon...
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan .venv banta.& set "VPY=C:\fake\python.exe"& goto :pip_stage)

if /i "%REINSTALL%"=="1" if exist "%ROOT%.venv" (
    echo        Purana .venv hata raha hoon - /reinstall diya gaya tha...
    rmdir /s /q "%ROOT%.venv"
)
if exist "%ROOT%.venv\Scripts\python.exe" goto :venv_ready
"%PYEXE%" -m venv "%ROOT%.venv" >>"%LOG%" 2>&1
if errorlevel 1 goto :err_venv

:venv_ready
set "VPY=%ROOT%.venv\Scripts\python.exe"
echo        [OK] Environment ready: .venv

REM ============================================================================
REM  [3/10]  Saare packages
REM ============================================================================
:pip_stage
echo.
echo  [3/10] Packages install ho rahe hain - ye sabse lamba step hai.
echo        PyQt6 (UI) + faster-whisper (voice) + chromadb (memory) +
echo        sentence-transformers + playwright + opencv + TTS/STT ... sab kuchh.
echo        Screen par kuchh hota hua na dikhe to ghabrana mat -
echo        progress setup_log.txt me ja raha hai.
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan pip install chalta.& goto :playwright_stage)

"%VPY%" -m pip install --upgrade pip setuptools wheel >>"%LOG%" 2>&1

echo        ... packages download + install (internet par depend) ...
"%VPY%" -m pip install -r "%ROOT%requirements.txt" --retries 5 --timeout 90 --prefer-binary >>"%LOG%" 2>&1
if not errorlevel 1 goto :pip_verify
echo        [WARN] Kuch packages fail hue - dobara try kar raha hoon...
"%VPY%" -m pip install -r "%ROOT%requirements.txt" --retries 5 --timeout 90 --prefer-binary --no-cache-dir >>"%LOG%" 2>&1
if not errorlevel 1 goto :pip_verify
echo        [WARN] Phir bhi kuch fail hue - zaroori packages alag alag install kar raha hoon...

set "WARNED=1"
REM sentencepiece + zstandard memory/conversation_db.py ke module-level import hain.
REM Inke bina app START nahi hota, isliye ye pehli (must-have) group me hain.
call :pip_group "sentencepiece zstandard PyQt6 PyQt6-WebEngine matplotlib markdown-it-py psutil pillow sounddevice numpy soundfile requests cryptography python-dotenv pygments loguru python-telegram-bot pynput prometheus-client"
call :pip_group "playwright beautifulsoup4 ddgs exa-py PyMuPDF pdfplumber pypdf pandas openpyxl python-docx python-pptx youtube-transcript-api"
call :pip_group "pyautogui pyperclip pygetwindow send2trash mss opencv-python onnxruntime plyer pydub miniaudio"
call :pip_group "faster-whisper edge-tts gTTS groq"
call :pip_group "chromadb sentence-transformers fastapi uvicorn pydantic msgpack"
call :pip_group "pywinauto comtypes uiautomation pycaw win10toast"
REM Hashim TTS (tools/gemini-live-tts) ke saare deps ek saath:
call :pip_group "google-genai pyaudio python-dotenv"
call :pip_pin "mcp>=1.27,<2"

:pip_verify
echo        Zaroori packages check kar raha hoon...
"%VPY%" -c "import sys,sentencepiece,zstandard,msgpack,PyQt6,numpy,psutil,requests,sounddevice,soundfile,PIL,cv2,chromadb,sentence_transformers,faster_whisper,edge_tts,playwright,bs4,loguru,dotenv,pygments,markdown_it,mcp;print('ok')" >>"%LOG%" 2>&1
if errorlevel 1 goto :pip_verify_fail
echo        [OK] Saare zaroori packages install ho gaye
goto :playwright_stage

:pip_verify_fail
echo        [WARN] Kuch zaroori packages abhi bhi missing hain.
echo               Orthos app pehli launch par apne aap bache hue install kar lega
echo               (lekin pehli launch thodi slow hogi). Details: setup_log.txt
set "WARNED=1"

REM ============================================================================
REM  [4/10]  Playwright Chromium
REM ============================================================================
:playwright_stage
echo.
echo  [4/10] Chromium browser download kar raha hoon (browser-control ke liye,
echo        ~150 MB). Ye sirf ek baar hota hai.
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan playwright install chalta.& goto :env_stage)

"%VPY%" -m playwright install chromium >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [WARN] Chromium download fail hua - internet check karke setup.bat dobara chalao.
    echo               Baaki Orthos chalega, sirf browser-control feature ko ye chahiye.
    set "WARNED=1"
) else (
    echo        [OK] Chromium ready
)

REM ============================================================================
REM  [5/10]  .env + config
REM ============================================================================
:env_stage
echo.
echo  [5/10] Config files check kar raha hoon...
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan .env banta.& goto :config_stage)
if exist "%ROOT%.env" goto :env_ready
if not exist "%ROOT%.env.example" goto :env_ready
copy /y "%ROOT%.env.example" "%ROOT%.env" >nul 2>&1
echo        [OK] .env ban gayi (.env.example se). API keys baad me Settings se daal sakte ho.

:env_ready

REM --- config/api_keys.json - working defaults --------------------------------
:config_stage
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan api_keys.json banta.& goto :config_ready)
if not exist "%ROOT%config" mkdir "%ROOT%config"
if exist "%ROOT%config\api_keys.json" goto :config_ready
echo        Default settings file bana raha hoon: config\api_keys.json
REM NOTE: stt_engine / tts_engine / llm_* yahan JAAN-BOOJH kar nahi likhe.
REM Orthos tab pehli launch par apna Setup window kholta hai (ui.py
REM _check_config) - usi me non-tech user se API key chuni jaati hai.
REM Yahan sirf wo defaults hain jo Setup form ke bahar hain (aur form unhe
REM merge karta hai, delete nahi).
> "%ROOT%config\api_keys.json" echo {
>> "%ROOT%config\api_keys.json" echo     "web_search_provider": "duckduckgo",
>> "%ROOT%config\api_keys.json" echo     "memory_search_backend": "local",
>> "%ROOT%config\api_keys.json" echo     "screen_vision_auto": false,
>> "%ROOT%config\api_keys.json" echo     "performance_mode": true,
>> "%ROOT%config\api_keys.json" echo     "performance_auto": true,
>> "%ROOT%config\api_keys.json" echo     "full_raw_history": false
>> "%ROOT%config\api_keys.json" echo }

"%VPY%" -c "import json,sys;json.load(open(sys.argv[1],encoding='utf-8'))" "%ROOT%config\api_keys.json" >>"%LOG%" 2>&1
if errorlevel 1 goto :config_bad
echo        [OK] config\api_keys.json ready
goto :config_ready

:config_bad
echo        [WARN] config\api_keys.json likhne me problem - app apne default use karega.

:config_ready

REM ============================================================================
REM  [6/10]  Runtime folders + MEMORY SYSTEM ka poora init
REM ============================================================================
:folders_stage
echo.
echo  [6/10] Runtime folders aur MEMORY SYSTEM taiyaar kar raha hoon...
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan folders + memory init + service test hote.& goto :tts_stage)

for %%D in ("%ROOT%logs" "%ROOT%models" "%ROOT%models\.cache" "%ROOT%captures" "%ROOT%memory\captures" "%ROOT%memory\compressed" "%ROOT%memory\chroma_db") do (
    if not exist "%%~D" mkdir "%%~D" >nul 2>&1
)
echo        [OK] logs / models / captures / memory folders ready

REM --- 6a: conversations.db (SQLite) init - tokenizer + schema ready ---------
echo        Memory database (SQLite) init kar raha hoon...
"%VPY%" -c "from memory.conversation_db import init_db; init_db(); print('memory-db-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [WARN] Memory DB init me dikkat - detail setup_log.txt me.
    echo               Pehli launch par app khud dobara try karega.
    set "WARNED=1"
) else (
    echo        [OK] Memory database ready: memory\conversations.db
)

REM --- 6b: ChromaDB vector store init ----------------------------------------
echo        Vector store (ChromaDB) init kar raha hoon...
"%VPY%" -c "import chromadb; c=chromadb.PersistentClient(path=r'memory\chroma_db'); c.get_or_create_collection('orthos_setup_check'); print('chroma-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [WARN] ChromaDB init me dikkat - detail setup_log.txt me.
    set "WARNED=1"
) else (
    echo        [OK] Vector store ready: memory\chroma_db
)

REM --- 6c: Memory Service (FastAPI, port 9876) LIVE smoke test ---------------
echo        Memory Service (FastAPI) ka live test kar raha hoon...
call :memory_service_smoke
if "%MSOK%"=="1" (
    echo        [OK] Memory Service chal rahi hai - http://127.0.0.1:9876/health OK
) else (
    echo        [WARN] Memory Service test me dikkat - detail setup_log.txt me.
    echo               Pehli launch par app khud isse start kar lega.
    set "WARNED=1"
)

REM ============================================================================
REM  [7/10]  TTS + STT + Hashim TTS verify
REM ============================================================================
:tts_stage
echo.
echo  [7/10] Voice engines (TTS/STT) check kar raha hoon...
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan TTS/STT verify + Hashim TTS deps hote.& goto :models_stage)

echo        Whisper STT + EdgeTTS + gTTS + Hashim TTS check...
"%VPY%" -c "import faster_whisper,edge_tts,gtts,groq;print('stt-tts-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [INFO] Kuch voice packages missing the - install kar raha hoon...
    call :pip_group "faster-whisper edge-tts gTTS groq"
)
echo        Hashim TTS deps (google-genai + pyaudio) check...
"%VPY%" -c "import google.genai,pyaudio,dotenv;print('hashim-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [INFO] Hashim TTS ke packages missing the - install kar raha hoon...
    call :pip_group "google-genai pyaudio python-dotenv"
)
REM tools/gemini-live-tts ka apna requirements bhi pakka (idempotent):
if exist "%ROOT%tools\gemini-live-tts\requirements.txt" (
    "%VPY%" -m pip install -r "%ROOT%tools\gemini-live-tts\requirements.txt" --retries 5 --timeout 90 --prefer-binary >>"%LOG%" 2>&1
)
"%VPY%" -c "import faster_whisper,edge_tts,gtts,groq,google.genai,pyaudio,dotenv;print('voice-all-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [WARN] Kuch voice packages abhi bhi missing - detail setup_log.txt me.
    echo               Orthos jo engine available hoga wahi use karega.
    set "WARNED=1"
) else (
    echo        [OK] Saare voice engines ready (Whisper + EdgeTTS + gTTS + Hashim TTS)
)

REM ============================================================================
REM  [8/10]  Models PRE-DOWNLOAD (pehli launch ko fast banata hai)
REM ============================================================================
:models_stage
echo.
echo  [8/10] Models pehle se download karoon? (ek hi baar hota hai)
echo        Whisper STT (~150 MB) + MiniLM embeddings (~90 MB) + Silero VAD (2 MB).
echo        Isse pehli launch 5-15 minute ki jagah ~1 minute me hogi.
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan models download hote.& goto :extras_check)
choice /C YN /T 25 /D Y /M "   Models download karein? [Y/N]"
if errorlevel 2 goto :models_skip

echo        [8a] Silero VAD download (2 MB)...
if exist "%ROOT%models\silero_vad.onnx" (
    echo        [OK] Silero VAD pehle se hai
) else (
    "%VPY%" -c "import urllib.request,pathlib; p=pathlib.Path(r'models\silero_vad.onnx'); p.parent.mkdir(exist_ok=True); urllib.request.urlretrieve('https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx', p); print('vad-ok')" >>"%LOG%" 2>&1
    if errorlevel 1 (
        echo        [WARN] VAD download nahi hua - app pehli launch par khud karega.
        set "WARNED=1"
    ) else (
        echo        [OK] Silero VAD ready
    )
)
echo        [8b] MiniLM embeddings download (~90 MB) - memory search ke liye...
"%VPY%" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('minilm-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [WARN] MiniLM download nahi hua - app pehli launch par khud karega.
    set "WARNED=1"
) else (
    echo        [OK] MiniLM embeddings ready
)
echo        [8c] Whisper 'base' STT download (~150 MB) - voice input ke liye...
"%VPY%" -c "from faster_whisper import WhisperModel; WhisperModel('base'); print('whisper-ok')" >>"%LOG%" 2>&1
if errorlevel 1 (
    echo        [WARN] Whisper download nahi hua - app pehli launch par khud karega.
    set "WARNED=1"
) else (
    echo        [OK] Whisper STT ready
)
goto :extras_check

:models_skip
echo        Theek hai - app inhe pehli launch par khud download kar lega (us din
echo        internet zaroori hoga, pehli launch slow hogi).

REM ============================================================================
REM  [9/10]  OPTIONAL extra tools
REM  Audit se nikla: MCP servers (config/mcp_servers.json) ke saare local
REM  servers "npx"/"uvx" use karte hain, audio convert ke liye "ffmpeg" hota
REM  hai, aur "ollama" optional offline LLM hai. Orthos inse bina bhi chalta
REM  hai (MCP failures handle hote hain, ffmpeg missing par message aata hai),
REM  isliye ye default OFF hain - UAC/permission prompt se bachne ke liye.
REM ============================================================================
:extras_check
echo.
echo  [9/10] Extra tools - ye OPTIONAL hain, Orthos inse bina bhi chalta hai:
echo        Node.js : MCP tools ke liye npx  (playwright / chrome / github MCP)
echo        uv      : kuchh MCP servers ke liye uvx
echo        Git     : repo aur version wale tools
echo        ffmpeg  : audio/video convert (pydub)
echo        Ollama  : PC par offline AI model - weak PC par ye SLOW hoga
echo        Windows permission (UAC) ka window aa sakta hai.
echo        Baad me kabhi bhi:  setup.bat /extras
if "%EXTRAS%"=="1" goto :extras_stage
choice /C YN /T 20 /D N /M "   Ye extra tools install karein? [Y/N]"
if errorlevel 2 goto :extras_done

:extras_stage
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan Node/uv/Git/ffmpeg/Ollama install hote.& goto :extras_done)
echo        Install ho raha hai (2-10 minute lag sakte hain)...
where winget >nul 2>&1
if errorlevel 1 goto :extras_nowinget
call :winget_try OpenJS.NodeJS.LTS "Node.js (npx)"
call :winget_try astral-sh.uv "uv (uvx)"
call :winget_try Git.Git "Git"
call :winget_try Gyan.FFmpeg "ffmpeg"
call :winget_try Ollama.Ollama "Ollama"
echo        [OK] Jo install ho paya wo ho gaya - detail setup_log.txt me.
goto :extras_done

:extras_nowinget
echo        [WARN] winget is PC par nahi mila - extras skip kar diye.
echo               Manual links: nodejs.org | docs.astral.sh/uv | git-scm.com
echo               ffmpeg.org | ollama.com

:extras_done
REM /extras-only run me ab seedha summary/exit (shortcut step skip).
if "%EXTRAS%"=="1" goto :summary

REM ============================================================================
REM  [10/10]  Shortcut + pehli launch
REM ============================================================================
:final_stage
echo.
echo  [10/10] Shortcut aur pehli launch...
if "%DRYRUN%"=="1" (echo        [DRY-RUN] Yahan shortcut + launch.& goto :summary)

choice /C YN /T 15 /D Y /M "   Desktop par Orthos ka shortcut banayein? [Y/N]"
if errorlevel 2 goto :skip_shortcut
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Orthos.lnk'); $s.TargetPath='%ROOT%start.bat'; $s.WorkingDirectory='%ROOT%'; $s.IconLocation='%SystemRoot%\System32\shell32.dll,21'; $s.Description='Orthos AI Desktop Assistant'; $s.Save()" >>"%LOG%" 2>&1
if exist "%USERPROFILE%\Desktop\Orthos.lnk" echo        [OK] Desktop shortcut ban gaya
goto :summary

:skip_shortcut
echo        Shortcut skip kiya - baad me start.bat se chala sakte ho.

:summary
echo.
echo  ==============================================================
if defined WARNED echo    SETUP POORA HUA  (kuchh warnings - neeche dekho)
if not defined WARNED echo    SETUP POORA HUA - SAB READY
echo  ==============================================================
echo.
echo   Orthos chalane ke liye:  is folder me  start.bat  double-click karo
if "%DRYRUN%"=="1" goto :dryrun_end
if defined WARNED echo.
if defined WARNED echo   Warnings the - poori detail "setup_log.txt" me hai. Orthos phir bhi
if defined WARNED echo   chalta hai; app pehli launch par bache packages khud install kar leta hai.
if defined WARNED echo.
echo   PEHLI LAUNCH PAR:
echo     - Ek Setup window khulega - usme apni API key daal dena, ek hi baar.
echo       Free key: Groq (console.groq.com) ya Gemini (aistudio.google.com).
echo     - Models pehle se download ho chuke hain, isliye launch jaldi hoga.
echo     - Memory system (SQLite + ChromaDB + Memory Service) aur voice engines
echo       (Whisper + EdgeTTS + Hashim TTS) sab ready hain.
echo     - Internet sirf pehli baar zaroori hai, uske baad sab offline chalta hai.
echo.
echo   Memory Service alag se chalani ho to:  start_memory_service.bat
echo   (Ye ab Orthos ke apne .venv se chalti hai - koi global Python nahi chahiye.)
echo.
echo   Poora offline dimaag chahiye to: https://ollama.com install karo, phir
echo   command chalao:  ollama pull qwen2.5:7b
echo.
echo   Extra tools (MCP browser tools ke liye Node.js, ffmpeg, Ollama) kabhi bhi
echo   install karne ke liye:   setup.bat /extras
echo.

choice /C YN /T 30 /D N /M "   Abhi Orthos start karein? [Y/N]"
if errorlevel 2 goto :bye
echo.
echo   Orthos start ho raha hai... (naya window khulega / logs dikhenge)
call "%ROOT%start.bat"
goto :bye

:dryrun_end
echo   [DRY-RUN] Kuchh bhi install nahi hua - sirf test tha. Sab kuchh theek hai.

:bye
if "%DRYRUN%"=="1" goto :bye_quiet
echo.
pause

:bye_quiet
endlocal
exit /b 0

REM ============================================================================
REM  SUBROUTINES
REM ============================================================================

REM -- :probe <exe-or-command> [launcher-arg] ----------------------------------
REM   Python dhoondhta hai. Valid = 3.11+, 64-bit, aur Windows Store ka fake
REM   stub nahi. Mila to PYEXE set karta hai, warna kuchh nahi.
:probe
if defined PYEXE exit /b 0
"%~1" %2 -c "import sys;raise SystemExit(0 if sys.version_info[:2]>=(3,11) and sys.maxsize>2**32 and 'WindowsApps' not in sys.executable else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
"%~1" %2 -c "import sys;print(sys.executable)" >"%TMPPY%" 2>nul
set /p PYEXE=<"%TMPPY%"
if not exist "%PYEXE%" set "PYEXE="
exit /b 0

REM -- :refresh_path -----------------------------------------------------------
REM   Naye install kiye Python ko isi window me available karta hai.
:refresh_path
set "PATH=%LocalAppData%\Programs\Python\Python312\;%LocalAppData%\Programs\Python\Python312\Scripts\;%LocalAppData%\Programs\Python\Python311\;%LocalAppData%\Programs\Python\Python311\Scripts\;%LocalAppData%\Programs\Python\Launcher\;%PATH%"
exit /b 0

REM -- :memory_service_smoke ---------------------------------------------------
REM   Memory Service (FastAPI, port 9876) ko background me start karke
REM   /health endpoint poll karta hai. Pass/fail MSOK me set hota hai.
REM   Service hamesha band kar di jaati hai - ye sirf EK TEST tha.
:memory_service_smoke
set "MSOK="
if "%DRYRUN%"=="1" exit /b 0
set "MSPID="
REM Launch python-wrapper se - PowerShell Start-Process -RedirectStandardOutput
REM pipe contexts me HANG hota hai (handle inheritance), isliye ye nahi.
REM Caret-escaped quotes ^" zaroori hain - warna cmd /c quotes strip kar deta hai.
for /f "delims=" %%P in ('^""%VPY%" -c "import subprocess,sys;f=open('logs/memory_service_setup.log','w');g=open('logs/memory_service_setup.err.log','w');p=subprocess.Popen([sys.executable,'-X','utf8','memory/memory_service.py'],stdout=f,stderr=g);print(p.pid)"^"') do set "MSPID=%%P"
if not defined MSPID (
    echo        [WARN] Memory Service start hi nahi ho payi - setup_log.txt dekho.
    exit /b 0
)
REM MiniLM + reranker bhi yahin load hote hain (pehli baar download ho sakte
REM hain) - isliye 90 x 3 = 4.5 minute tak intezaar. Poll curl.exe se
REM (har iteration me PowerShell spawn karna bahut slow hota hai).
for /l %%I in (1,1,90) do (
    if not defined MSOK (
        curl -s -m 5 http://127.0.0.1:9876/health >nul 2>&1
        if not errorlevel 1 set "MSOK=1"
        if not defined MSOK ping -n 4 127.0.0.1 >nul
    )
)
if defined MSPID powershell -NoProfile -Command "Stop-Process -Id %MSPID% -Force -ErrorAction SilentlyContinue" >nul 2>&1
if "%MSOK%"=="1" (
    echo        [OK] Memory Service live test PASS (port 9876)
) else (
    echo        [INFO] Memory Service /health tak nahi pahunchi - log:
    type "%ROOT%logs\memory_service_setup.err.log" >>"%LOG%" 2>&1
)
exit /b 0

REM -- :pip_group "<space separated packages>" ---------------------------------
:pip_group
if "%DRYRUN%"=="1" exit /b 0
echo          - %~1
"%VPY%" -m pip install %~1 --retries 5 --timeout 90 --prefer-binary >>"%LOG%" 2>&1
exit /b 0

REM -- :pip_pin "<single package with version range>" -------------------------
:pip_pin
if "%DRYRUN%"=="1" exit /b 0
echo          - %~1
"%VPY%" -m pip install "%~1" --retries 5 --timeout 90 --prefer-binary >>"%LOG%" 2>&1
exit /b 0

REM -- :winget_try <package-id> <label> ---------------------------------------
REM    Optional tools: fail hone par sirf warning, setup nahi rukta.
:winget_try
if "%DRYRUN%"=="1" exit /b 0
echo          - %~2
winget install --id %~1 --exact --silent --accept-package-agreements --accept-source-agreements >>"%LOG%" 2>&1
if errorlevel 1 echo          [WARN] %~2 install nahi ho paya - permission ya network issue.
exit /b 0

REM -- Errors ------------------------------------------------------------------

:err_wrong_folder
echo.
echo  [ERROR] setup.bat Orthos ke folder me nahi hai.
echo          setup.bat ko wahan rakho jahan main.py aur requirements.txt hain,
echo          phir dobara double-click karo.
goto :err_exit

:err_venv
echo.
echo  [ERROR] .venv nahi ban paya - detail: setup_log.txt
echo          Aksar antivirus ya folder permission ki wajah se hota hai.
echo          Folder ko Desktop par extract karke dobara try karo.
goto :err_exit

:err_exit
echo.
pause
endlocal
exit /b 1

:usage
echo.
echo   Orthos setup - saare commands:
echo.
echo     setup.bat              Normal setup (safe, dobara chalane par skip karta hai)
echo     setup.bat /reinstall   .venv hata kar poora fresh install
echo     setup.bat /dryrun      Sirf test - kuchh install nahi karta
echo     setup.bat /extras      Sirf optional extra tools (Node/uv/Git/ffmpeg/Ollama)
echo.
echo     Setup me sab kuchh hota hai: packages, Chromium, memory system
echo     (SQLite + ChromaDB + FastAPI Memory Service live test), voice engines
echo     (Whisper STT + EdgeTTS + gTTS + Hashim TTS), aur model pre-download.
echo.
pause
endlocal
exit /b 0
