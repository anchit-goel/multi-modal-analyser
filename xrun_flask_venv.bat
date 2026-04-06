@echo off
cd /d %~dp0
start "" /B "%~dp0.venv\Scripts\python.exe" app.py
