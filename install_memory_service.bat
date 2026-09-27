@echo off
title Install Orthos Memory Service (Task Scheduler)
echo ============================================
echo  Orthos - Memory Service Installer
echo ============================================
echo.
echo Requires Administrator privileges.
echo.

REM Navigate to this script's directory (portable - works from any location)
cd /d "%~dp0"

REM Get the full path to this project folder
set "PROJECT_DIR=%~dp0"
REM Remove trailing backslash
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

REM Find Python automatically
set "PYTHON_PATH=python"
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Install Python 3.11+ and add to PATH.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('where python') do (
    set "PYTHON_PATH=%%i"
    goto :found_python
)
:found_python
echo Using Python: %PYTHON_PATH%
echo Project directory: %PROJECT_DIR%
echo.

REM Delete old task if exists
schtasks /delete /tn "Orthos-MemoryService" /f 2>nul

REM Create new task: runs at boot
schtasks /create /tn "Orthos-MemoryService" ^
    /tr "\"%PYTHON_PATH%\" \"%PROJECT_DIR%\memory\memory_service.py\"" ^
    /sc onstart ^
    /ru "%USERNAME%" ^
    /rl limited ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Task created: Orthos-MemoryService
    echo It will run automatically at every PC boot.
    echo ChromaDB vector search will be pre-loaded.
    echo.
    echo To start it now without rebooting:
    echo   schtasks /run /tn "Orthos-MemoryService"
    echo.
    echo To remove:
    echo   schtasks /delete /tn "Orthos-MemoryService" /f
) else (
    echo.
    echo [ERROR] Failed to create task. Run as Administrator.
)

echo.
pause
