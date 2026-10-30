param(
    [ValidateSet('full', 'native')][string]$Model = 'full',
    [ValidateRange(1,65535)][int]$Port = 8080,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$selectionPath = Join-Path $repoRoot 'results/realsense_20261029/selection.json'
$cameraIndex = 0
if ($Model -eq 'full' -and (Test-Path -LiteralPath $selectionPath)) {
    $selected = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
    $checkpoint = $selected.checkpoint
    $dataDir = $selected.dataset
    if ($null -ne $selected.viewer_camera_index) { $cameraIndex = $selected.viewer_camera_index }
} elseif ($Model -eq 'native') {
    $checkpoint = 'results/realsense_20261029/global_native_depth15k/ckpts/final_post_update.pt'
    $dataDir = 'data/realsense_20261029_140739/global_native_dataset'
    $cameraIndex = 150
} else {
    $checkpoint = 'results/realsense_20261029/full_scene_626_depth30k/ckpts/final_post_update.pt'
    $dataDir = 'data/realsense_20261029_140739/rgbd/dataset'
}
& "$PSScriptRoot/view-room.ps1" -Checkpoint $checkpoint -DataDir $dataDir -CameraIndex $cameraIndex -Port $Port -Background:$Background -OutputDir 'results/realsense_20261029/viewer'
