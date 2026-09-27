@echo off
setlocal enabledelayedexpansion
title Orthos - Memory Service Mode Switcher (Local ^<-> Cloud)

echo ============================================
echo  Orthos - Memory Service Mode
echo ============================================
echo.
echo   [1] LOCAL   - i3 PC pe memory service (start_memory_service.bat se chalta hai)
echo   [2] CLOUD   - Lightning AI Studio wali fast memory (Tailscale URL chahiye)
echo   [3] STATUS  - current mode + reachability check
echo.
choice /C 123 /N /M "Select 1, 2 or 3: "
set _sel=%errorlevel%

REM ── helpers ──────────────────────────────────────────────────────────────
set "CFG=%~dp0config\api_keys.json"

if %_sel%==1 goto local
if %_sel%==2 goto cloud
goto status

:local
echo.
echo Switching to LOCAL memory service...
call :set_url ""
echo [OK] memory_service_url cleared - local service (127.0.0.1:9876) active.
echo.
echo Start the local service now?
choice /C YN /N /M "Run start_memory_service.bat? [Y/N]: "
if !errorlevel!==1 (
    start "" "%~dp0start_memory_service.bat"
)
goto done

:cloud
echo.
set /p RAWURL="Tailscale IP or full URL (e.g. 100.x.y.z  ya  http://100.x.y.z:9876): "
if "%RAWURL%"=="" (
    echo [X] URL required.
    goto done
)
set /p RAWTOKEN="Memory token (ENTER to skip if service has no MEMORY_TOKEN): "

REM ── trim + normalize (spaces, http://, :9876) inside PowerShell ──
REM Rule: no scheme typed -> bare IP/host: add http:// and :9876 (unless a port given).
REM       full URL typed -> trust as-is.
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$raw = '%RAWURL%'.Trim(); if ($raw -and ($raw -notmatch '://')) { $u = 'http://' + $raw; if ($raw -notmatch ':\d+$') { $u = $u + ':9876' } } else { $u = $raw }; $u"`) do set "TSURL=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "('%RAWTOKEN%').Trim()"`) do set "TOKEN=%%i"

if "!TSURL!"=="" (
    echo [X] URL required.
    goto done
)
echo Using: !TSURL!

echo.
echo Saving to config...
call :set_url "!TSURL!"
if not "!TOKEN!"=="" call :set_token "!TOKEN!"

echo.
echo Testing connection...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '!TSURL!/health' -Headers @{Authorization='Bearer !TOKEN!'} -TimeoutSec 8 -UseBasicParsing; Write-Host '[OK] Cloud memory service reachable:' $r.Content } catch { Write-Host '[X] UNREACHABLE -' $_.Exception.Message; Write-Host '     Check: Tailscale connected? Studio running? watchdog.sh alive?' }"
echo.
echo [DONE] Orthos ab CLOUD memory use karega. Restart not needed.
echo        Local fallback: agar cloud down ho aur local service chal rahi
echo        ho to Orthos automatic wahan switch kar lega.
goto done

:status
echo.
echo Current config:
powershell -NoProfile -Command "$cfg = Get-Content '!CFG!' -Raw | ConvertFrom-Json; $u = $cfg.memory_service_url; if ([string]::IsNullOrWhiteSpace($u)) { Write-Host '  mode  : LOCAL (127.0.0.1:9876)' } else { Write-Host '  mode  : CLOUD'; Write-Host ('  url   : ' + $u) }; if ($cfg.memory_service_token) { Write-Host '  token : set' }"
echo.
echo Reachability:
powershell -NoProfile -Command "$cfg = Get-Content '!CFG!' -Raw | ConvertFrom-Json; $u = $cfg.memory_service_url; $hdr = @{}; if ($cfg.memory_service_token) { $hdr.Authorization = 'Bearer ' + $cfg.memory_service_token }; if ([string]::IsNullOrWhiteSpace($u)) { $u = 'http://127.0.0.1:9876' }; try { $r = Invoke-WebRequest -Uri ($u + '/health') -Headers $hdr -TimeoutSec 5 -UseBasicParsing; Write-Host ('  [OK] ' + $u + ' -> ' + $r.Content) } catch { Write-Host ('  [X] ' + $u + ' unreachable') }; try { $r2 = Invoke-WebRequest -Uri 'http://127.0.0.1:9876/health' -TimeoutSec 2 -UseBasicParsing; Write-Host ('  [OK] local service running') } catch { Write-Host '  [--] local service not running' }"
goto done

:set_url
REM Write WITHOUT BOM: Out-File default UTF8 adds a BOM that breaks Python's json.load
powershell -NoProfile -Command "$f='!CFG!'; $cfg = Get-Content $f -Raw | ConvertFrom-Json; if ('%~1' -eq '') { $cfg.PSObject.Properties.Remove('memory_service_url') } else { $cfg | Add-Member -Force -NotePropertyName memory_service_url -NotePropertyValue '%~1' }; $json = $cfg | ConvertTo-Json -Depth 10; [IO.File]::WriteAllText($f, $json, (New-Object System.Text.UTF8Encoding($false)))"
exit /b

:set_token
powershell -NoProfile -Command "$f='!CFG!'; $cfg = Get-Content $f -Raw | ConvertFrom-Json; $cfg | Add-Member -Force -NotePropertyName memory_service_token -NotePropertyValue '%~1'; $json = $cfg | ConvertTo-Json -Depth 10; [IO.File]::WriteAllText($f, $json, (New-Object System.Text.UTF8Encoding($false)))"
exit /b

:done
echo.
pause
