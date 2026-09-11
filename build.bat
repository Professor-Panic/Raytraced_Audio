@echo off
cd /d "%~dp0"
cmake --build build --target run
pause