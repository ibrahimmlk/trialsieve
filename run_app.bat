@echo off
REM TrialSieve: double-click to start the browser application.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the Python environment and installing dependencies...
    python -m venv .venv || goto :error
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error
    .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
echo Starting TrialSieve. The browser opens automatically at http://127.0.0.1:7860 once the models are loaded.
.venv\Scripts\python app\app.py
if errorlevel 1 goto :error
goto :eof
:error
echo.
echo TrialSieve failed to start. See the messages above.
pause
