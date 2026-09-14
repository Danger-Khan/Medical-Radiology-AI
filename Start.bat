@echo off
REM Medical Radiology AI (Desktop) launcher.
REM First run downloads real model weights from Hugging Face (a few hundred MB) and installs
REM Python deps if missing -- needs internet once. After that, radiology_console.py runs fully offline.
REM All the actual code/data lives in App\ -- this launcher just cd's in and runs it.

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
    echo Dependency install failed. Check your internet connection and re-run Start.bat.
    pause
    exit /b 1
)

echo Launching Medical Radiology AI...
python radiology_console.py

if errorlevel 1 (
    echo.
    echo Medical Radiology AI exited with an error. See the message above.
    pause
)
