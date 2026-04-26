$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "New Trade Dashboard.lnk"
$exePath = Join-Path $ProjectRoot "dist\NewTradeDashboard\NewTradeDashboard.exe"
$fallbackTarget = Join-Path $ProjectRoot "start_dashboard.bat"
$iconPath = Join-Path $ProjectRoot "assets\new_trade_dashboard.ico"

$targetPath = if (Test-Path $exePath) { $exePath } else { $fallbackTarget }

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.IconLocation = $iconPath
$shortcut.Description = "Open New Trade Dashboard"
$shortcut.Save()

Write-Host "[OK] Desktop shortcut updated"
Write-Host "[PATH] $shortcutPath"
Write-Host "[TARGET] $targetPath"
