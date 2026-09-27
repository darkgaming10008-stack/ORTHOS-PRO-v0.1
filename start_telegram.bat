@echo off
title Orthos - Telegram Bot
echo ============================================
echo  Orthos - Telegram Bot (Headless)
echo ============================================
echo.

REM Navigate to this script's directory (portable - works from any location)
cd /d "%~dp0"

echo Stopping old instances...
wmic process where "name='python.exe' and CommandLine like '%%main.py%%'" call terminate 2>nul
timeout /t 2 /nobreak >nul

echo [%date% %time%] Starting Orthos Telegram bot (headless)...
python main.py --headless

echo.
echo [%date% %time%] Bot stopped.
pause
