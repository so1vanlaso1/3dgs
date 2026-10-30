param(
    [ValidateRange(1,65535)][int]$Port = 8082,
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$reportPath = Join-Path $repoRoot 'results/realsense_floater_cleanup_20261030/report.json'
if (-not (Test-Path -LiteralPath $reportPath)) { throw 'Run scripts/clean-realsense-floaters.py first.' }
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
& "$PSScriptRoot/view-room.ps1" -Checkpoint $report.viewer.checkpoint -DataDir $report.viewer.dataset `
    -CameraIndex $report.viewer.camera_index -Port $Port -Background:$Background `
    -OutputDir 'results/realsense_floater_cleanup_20261030/viewer'
