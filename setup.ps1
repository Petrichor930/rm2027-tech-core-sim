# 一键安装（PowerShell 版，功能同 setup.bat）：自动查找 Python -> venv -> 装依赖 -> 自检
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cands = @()
$cmd = Get-Command python -ErrorAction SilentlyContinue
if ($cmd) { $cands += $cmd.Source }
try { $cands += (py -3 -c "import sys;print(sys.executable)") } catch {}
$cands += (Get-ChildItem "$env:LocalAppData\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue).FullName
$cands += (Get-ChildItem "$env:ProgramFiles\Python3*\python.exe" -ErrorAction SilentlyContinue).FullName
$cands += (Get-ChildItem "C:\Python3*\python.exe" -ErrorAction SilentlyContinue).FullName
foreach ($d in @('C:', 'D:', 'E:', 'F:', 'G:', 'H:')) {
    $cands += (Get-ChildItem "$d\*\python.exe" -ErrorAction SilentlyContinue).FullName
    if (Test-Path "$d\python.exe") { $cands += "$d\python.exe" }
}

$ranked = @()
foreach ($p in ($cands | Where-Object { $_ } | Select-Object -Unique)) {
    if (-not (Test-Path $p)) { continue }
    if ($p -match "msys64|cygwin|WindowsApps") { continue }
    & $p -c "import sys" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { continue }
    $v = [int](& $p -c "import sys;print(sys.version_info[0]*100+sys.version_info[1])")
    if ($v -ge 308) { $ranked += [pscustomobject]@{ V = $v; P = $p } }
}
# 不按版本优先：保持发现顺序，先找到的能用就用
if (-not $ranked) { Write-Host "[错误] 未找到可用的 Windows Python 3.8+。"; exit 1 }

$pyok = $null
foreach ($r in $ranked) {
    Write-Host "[尝试] $($r.P)"
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
    & $r.P -m venv .venv 2>$null | Out-Null
    if (Test-Path ".venv/Scripts/python.exe") { $pyok = $r.P; break }
    Write-Host "[警告] 该解释器的 venv 布局不兼容，换下一个"
    if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
}
if (-not $pyok) { Write-Host "[错误] 所有候选的 venv 均不可用。"; exit 1 }
Write-Host "[使用 Python] $pyok"
& $pyok --version

Write-Host "[2/3] 安装依赖 ..."
& "./.venv/Scripts/python.exe" -m pip install --upgrade pip -q
& "./.venv/Scripts/python.exe" -m pip install -r requirements.txt -q

Write-Host "[3/3] 自检（无头验证）..."
& "./.venv/Scripts/python.exe" src/verify_steps.py

Write-Host ""
Write-Host "================= 安装完成 ================="
Write-Host "运行仿真:  run.bat   （或 .venv/Scripts/python.exe src/assemble_view.py）"
