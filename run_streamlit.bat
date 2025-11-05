@echo off
rem Run Streamlit app and open default browser to the UI
cd /d "%~dp0"

rem Start Streamlit in a new window (so this script can continue)
start "Streamlit" cmd /c "python -m streamlit run streamlit_app.py"

rem Wait a moment for server to start, then open default browser to the app URL
timeout /t 3 /nobreak >nul
start "" "http://localhost:8501"

rem Note: If your environment requires activating a venv, modify this script to activate it before running Streamlit.