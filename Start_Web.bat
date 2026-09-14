@echo off
REM Medical Radiology AI (Streamlit web edition) launcher — opens a local browser UI, same real
REM models and SQLite cache as the desktop console (Start.bat / radiology_console.py). First run
REM needs internet once to download model weights + Python deps; after that it's a local-only web
REM server (no cloud calls). All the actual code/data lives in App\ -- this launcher just cd's in
REM and runs it.

cd /d "%~dp0App"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from https://python.org and try again.
    pause
    exit /b 1
)

echo Checking Python dependencies (this only installs anything missing)...
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed. Check your internet connection and re-run Start_Web.bat.
    pause
    exit /b 1
)

echo Launching Medical Radiology AI (Streamlit)...
python -m streamlit run radiology_web.py

if errorlevel 1 (
    echo.
    echo Medical Radiology AI exited with an error. See the message above.
    pause
)
