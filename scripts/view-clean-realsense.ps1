param(
    [ValidateRange(1,65535)][int]$Port = 8083,
    [switch]$Background,
    [string]$ResultDir = 'results/realsense_20261030_111354_clean'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$selected = Get-Content -LiteralPath (Join-Path $repoRoot "$ResultDir/viewer.json") -Raw | ConvertFrom-Json
& "$PSScriptRoot/view-room.ps1" -Checkpoint $selected.checkpoint -DataDir $selected.dataset `
    -CameraIndex $selected.camera_index -Port $Port -Background:$Background `
    -OutputDir "$ResultDir/viewer"
