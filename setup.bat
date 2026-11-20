@echo off
rem 一键安装：自动查找 Python（排除 MSYS/商店占位）-> 创建虚拟环境 -> 装依赖 -> 自检
chcp 65001 >nul
cd /d "%~dp0"
setlocal EnableDelayedExpansion

set "CANDFILE=%TEMP%\_pycand.txt"
if exist "%CANDFILE%" del "%CANDFILE%"

rem --- 1) PATH 中的 python ---
for /f "delims=" %%W in ('where python 2^>nul') do call :try "%%W"

rem --- 2) py 启动器（解析实际路径）---
for /f "delims=" %%W in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do call :try "%%W"

rem --- 3) 常见安装目录 ---
for /d %%P in ("%LocalAppData%\Programs\Python\Python3*" "%ProgramFiles%\Python3*" "C:\Python3*") do call :try "%%~P\python.exe"

rem --- 4) 各盘符根目录一级（便携版）---
for %%D in (C D E F G H) do (
    for /d %%P in ("%%D:\*") do call :try "%%~P\python.exe"
    call :try "%%D:\python.exe"
)

rem --- 依发现顺序尝试，先找到的能用就用（不按版本优先）---
if not exist "%CANDFILE%" (
    echo [错误] 未找到可用的 Windows Python 3.8+。请安装 Python 后重试。
    pause
    exit /b 1
)

rem --- 按版本从高到低尝试；venv 布局不兼容则换下一候选 ---
set "PYOK="
for /f "usebackq tokens=1* delims=|" %%A in ("%CANDFILE%") do (
    if not defined PYOK (
        echo [尝试] %%B
        if exist .venv rmdir /s /q .venv
        "%%B" -m venv .venv >nul 2>nul
        if exist ".venv/Scripts/python.exe" (
            set "PYOK=%%B"
        ) else (
            echo [警告] 该解释器的 venv 布局不兼容，换下一个
            if exist .venv rmdir /s /q .venv
        )
    )
)
if not defined PYOK (
    echo [错误] 所有候选的 venv 均不可用。
    pause
    exit /b 1
)
echo [使用 Python] %PYOK%
"%PYOK%" --version

echo [2/3] 安装依赖 ...
".venv/Scripts/python.exe" -m pip install --upgrade pip -q
".venv/Scripts/python.exe" -m pip install -r requirements.txt -q || (echo [错误] 依赖安装失败 & pause & exit /b 1)

echo [3/3] 自检（无头验证）...
".venv/Scripts/python.exe" src/verify_steps.py || (echo [错误] 自检失败 & pause & exit /b 1)

echo.
echo ================= 安装完成 =================
echo 运行仿真:  run.bat   （或 .venv/Scripts/python.exe src/assemble_view.py）
pause
goto :eof

rem ===== 候选收集：可用、非 MSYS、版本>=3.8 则记入列表（版本|路径）=====
:try
set "CAND=%~1"
if not defined CAND goto :eof
if not exist "%CAND%" goto :eof
echo %CAND% | findstr /i "msys64 cygwin WindowsApps" >nul
if not errorlevel 1 goto :eof
"%CAND%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :eof
set "VER="
"%CAND%" -c "import sys;print(sys.version_info[0]*100+sys.version_info[1])" > "%TEMP%\_pyver.txt" 2>nul
set /p VER=<"%TEMP%\_pyver.txt"
if not defined VER goto :eof
if !VER! LSS 308 goto :eof
set "PAD=0000!VER!"
set "PAD=!PAD:~-4!"
>> "%CANDFILE%" echo !PAD!^|%CAND%
goto :eof
