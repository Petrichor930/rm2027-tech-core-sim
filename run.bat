@echo off
rem 启动交互仿真（先跑 setup.bat 安装）
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv/Scripts/python.exe (echo [错误] 未安装，请先运行 setup.bat & pause & exit /b 1)
".venv/Scripts/python.exe" src/assemble_view.py
