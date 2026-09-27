@echo off
title Orthos - Starting...
echo ============================================
echo  Orthos - AI Desktop Assistant
echo ============================================
echo.

REM Navigate to this script's directory (portable - works from any location)
cd /d "%~dp0"

echo Stopping old instances...
wmic process where "name='python.exe' and CommandLine like '%%main.py%%'" call terminate 2>nul
taskkill /F /IM "main.exe" 2>nul
timeout /t 2 /nobreak >nul

echo Starting Orthos...
start "" python main.py
echo.
echo Orthos is running. Close this window.
