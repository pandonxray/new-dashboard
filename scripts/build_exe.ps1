$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
  $pythonPath = "D:\codex\Excel-\trade_dashboard\.venv\Scripts\python.exe"
}

$distPath = Join-Path $ProjectRoot "dist"
$buildPath = Join-Path $ProjectRoot "build"
$iconPath = Join-Path $ProjectRoot "assets\new_trade_dashboard.ico"

if (Test-Path $distPath) { Remove-Item $distPath -Recurse -Force }
if (Test-Path $buildPath) { Remove-Item $buildPath -Recurse -Force }

& $pythonPath -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --name NewTradeDashboard `
  --collect-all streamlit `
  --collect-all plotly `
  --collect-all pyarrow `
  --hidden-import pandas `
  --hidden-import openpyxl `
  --hidden-import yaml `
  --icon $iconPath `
  --add-data "config;config" `
  --add-data "src;src" `
  src\launcher.py

Write-Host ""
Write-Host "[OK] Build complete"
Write-Host "[PATH] $ProjectRoot\dist\NewTradeDashboard\NewTradeDashboard.exe"
