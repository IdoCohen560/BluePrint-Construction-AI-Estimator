@echo off
cd /d "%~dp0"
echo Starting Flask app (use this if Streamlit shows error -102).
echo Keep this window OPEN.
echo Open: http://127.0.0.1:8765
echo.
python web_flask.py
echo.
pause >nul
