# 一键安装（PowerShell 版，功能同 setup.bat）
# 用法: powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[错误] 未找到 python，请先安装 Python 3.9+ 并加入 PATH"; exit 1
}

if (-not (Test-Path .venv)) {
    Write-Host "[1/3] 创建虚拟环境 .venv ..."
    python -m venv .venv
} else {
    Write-Host "[1/3] .venv 已存在，跳过创建"
}

Write-Host "[2/3] 安装依赖 ..."
& "./.venv/Scripts/python.exe" -m pip install --upgrade pip -q
& "./.venv/Scripts/python.exe" -m pip install -r requirements.txt -q

Write-Host "[3/3] 自检（无头验证）..."
& "./.venv/Scripts/python.exe" src/verify_steps.py

Write-Host ""
Write-Host "================= 安装完成 ================="
Write-Host "运行仿真:  run.bat   （或 .venv/Scripts/python.exe src/assemble_view.py）"
