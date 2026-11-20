@echo off
rem 一键安装：创建虚拟环境 -> 装依赖 -> 自检
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul || (echo [错误] 未找到 python，请先安装 Python 3.9+ 并加入 PATH & pause & exit /b 1)

if not exist .venv (
    echo [1/3] 创建虚拟环境 .venv ...
    python -m venv .venv || (echo [错误] venv 创建失败 & pause & exit /b 1)
) else (
    echo [1/3] .venv 已存在，跳过创建
)

echo [2/3] 安装依赖 ...
".venv/Scripts/python.exe" -m pip install --upgrade pip -q
".venv/Scripts/python.exe" -m pip install -r requirements.txt -q || (echo [错误] 依赖安装失败 & pause & exit /b 1)

echo [3/3] 自检（无头验证）...
".venv/Scripts/python.exe" src/verify_steps.py || (echo [错误] 自检失败 & pause & exit /b 1)

echo.
echo ================= 安装完成 =================
echo 运行仿真:  run.bat   （或 .venv/Scripts/python.exe src/assemble_view.py）
pause
