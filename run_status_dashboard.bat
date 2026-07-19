@echo off
setlocal
cd /d "%~dp0"

REM Prefer Miniconda (user machine); fall back to conda base or PATH python.
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

echo Using Python: %PY%
echo Installing Streamlit deps if needed...
"%PY%" -m pip install -q -r requirements_streamlit.txt
if errorlevel 1 (
  echo ERROR: pip install failed.
  pause
  exit /b 1
)

echo.
echo Starting LapAI status dashboard...
echo Keep this window open. Browser: http://localhost:8501
echo.
"%PY%" -m streamlit run streamlit_status.py --server.headless true --browser.gatherUsageStats false %*
if errorlevel 1 (
  echo ERROR: Streamlit exited with an error.
  pause
  exit /b 1
)
