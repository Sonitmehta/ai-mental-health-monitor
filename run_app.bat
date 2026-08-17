@echo off
title MindScan AI - Mental Health Monitor
echo ====================================================
echo        Starting MindScan AI Web Application
echo ====================================================
echo.
cd /d "%~dp0"
echo Activating Virtual Environment...
call .venv\Scripts\activate.bat
echo Starting Server on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python app.py
pause
