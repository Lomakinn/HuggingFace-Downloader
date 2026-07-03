@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment for HuggingFace Downloader v1.01...
    py -3 -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b 1
    )

    echo Installing dependencies...
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to upgrade pip.
        pause
        exit /b 1
    )

    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Starting HuggingFace Downloader v1.01...
"venv\Scripts\python.exe" hf_downloader.py
if errorlevel 1 (
    echo Application closed with an error.
    pause
    exit /b 1
)

endlocal
