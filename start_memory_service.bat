@echo off
title Orthos - Memory Service (ChromaDB + SQLite)
echo ============================================
echo  Orthos - Memory Service
echo ============================================
echo.

REM Navigate to this script's directory (portable - works from any location)
cd /d "%~dp0"

echo Stopping old memory service instance...
wmic process where "name='python.exe' and CommandLine like '%%memory_service.py%%'" call terminate 2>nul
timeout /t 1 /nobreak >nul

echo Starting Orthos Memory Service...
python memory\memory_service.py
echo.
echo Memory service stopped.
pause
