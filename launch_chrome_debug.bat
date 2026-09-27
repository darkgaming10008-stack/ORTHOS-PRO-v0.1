@echo off
title Chrome Debug (Port 9222) - Orthos
echo.
echo  ============================================
echo   Chrome Remote Debug Launch
echo   Connects existing browser to Orthos via CDP
echo  ============================================
echo.

set /p choice="[1] Close all Chrome ^& launch new debug instance  [2] Just start Chrome normally (enable debug from chrome://inspect)  [default: 1]: "

if "%choice%"=="2" goto normal
goto debug

:debug
echo Closing existing Chrome to avoid profile lock conflicts...
taskkill /F /IM chrome.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo.
echo Launching Chrome with remote debugging port 9222 (dedicated profile)...
echo Your current browsing session will NOT be restored.
echo A fresh Chrome window with your extensions will open.
echo.
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Temp\chrome_debug_profile" --no-first-run --new-window "about:blank"
echo.
echo  Chrome started on port 9222
echo  Orthos's Playwright MCP will now connect to THIS browser
echo  You can log in to sites here - sessions will persist
echo.
goto end

:normal
echo.
echo  Open Chrome normally, then go to:
echo     chrome://inspect/#remote-debugging
echo     and enable "Allow remote debugging for this browser instance"
echo.
echo  Then start Orthos and it will auto-connect via CDP.
echo.

:end
echo.
echo  Tip: Bookmark chrome://inspect/#remote-debugging for quick access
echo  Tip: Your Chrome profile will persist in C:\Temp\chrome_debug_profile
pause
