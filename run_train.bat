@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
%PYTHON% src\eda.py
if errorlevel 1 exit /b 1
%PYTHON% src\train.py %*
exit /b %errorlevel%
