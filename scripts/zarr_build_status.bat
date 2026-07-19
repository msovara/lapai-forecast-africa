@echo off
REM One-shot Zarr build status (no loop).
cd /d "%~dp0.."
set ZARR=data\processed\lapai\era5_n96_2020_2021.zarr
set LOG=reports\zarr_build_debug.log
if not exist "%LOG%" set LOG=reports\zarr_build_stderr.log
echo === Zarr build status ===
echo Zarr: %ZARR%
if exist "%ZARR%\data" (
  for /f %%a in ('dir /b /a-d "%ZARR%\data" 2^>nul ^| find /c /v ""') do set CHUNKS=%%a
) else (
  set CHUNKS=0
)
powershell -NoProfile -Command "$m=(Get-ChildItem '%ZARR%' -Recurse -File -EA SilentlyContinue|Measure Length -Sum).Sum; Write-Host ('data_chunks={0} size_mb={1}' -f $env:CHUNKS,[math]::Round($m/1MB,1))"
tasklist /FI "IMAGENAME eq anemoi-datasets.exe" 2>nul | find /I "anemoi-datasets" >nul && echo build_process=running || echo build_process=not_running
if exist "%LOG%" (
  echo --- last log line ---
  powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 1"
) else (
  echo no log file yet
)
