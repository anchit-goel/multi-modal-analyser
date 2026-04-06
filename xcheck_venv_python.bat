@echo off
"%~dp0.venv\Scripts\python.exe" -c "import sys; print(sys.executable); import torch; print(torch.__file__)"
