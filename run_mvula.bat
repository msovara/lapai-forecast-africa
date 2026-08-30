@echo off
setlocal
cd /d "%~dp0"

REM Prefer Miniconda; fall back to Anaconda or PATH python.
set "PY="
if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo ERROR: Python not found. Install Miniconda or add python to PATH.
  pause
  exit /b 1
)

if "%~1"=="" (
  "%PY%" run_mvula.py help
  exit /b 0
)

"%PY%" run_mvula.py %*
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" pause
exit /b %EC%
