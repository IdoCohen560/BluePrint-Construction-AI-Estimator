@echo off
cd /d "%~dp0"
echo Starting Streamlit... Keep this window OPEN while you use the browser.
echo.
echo Try these URLs:
echo   http://127.0.0.1:8501
echo   http://localhost:8501
echo.
python -m streamlit run app.py
echo.
echo Server stopped. Press any key to close.
pause >nul
