param(
    [ValidateRange(1,65535)][int]$Port = 8081,
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$selectionPath = Join-Path $repoRoot 'results/realsense_15fps_experiment/viewer.json'
if (-not (Test-Path -LiteralPath $selectionPath)) {
    throw 'The 15 FPS experiment has not finished comparison and verification yet.'
}
$selected = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
& "$PSScriptRoot/view-room.ps1" -Checkpoint $selected.checkpoint -DataDir $selected.dataset `
    -CameraIndex $selected.camera_index -Port $Port -Background:$Background `
    -OutputDir 'results/realsense_15fps_experiment/viewer'
