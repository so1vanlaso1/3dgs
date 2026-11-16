param(
    [ValidateRange(1,65535)][int]$Port = 8085,
    [switch]$Background,
    [ValidateSet('selected','baseline','refined')][string]$Variant = 'selected'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$resultDir = 'results/realsense_small_pose_20261030'
if ($Variant -ne 'selected') { $resultDir = "$resultDir/$Variant" }
$selected = Get-Content -LiteralPath (Join-Path $repoRoot "$resultDir/viewer.json") -Raw | ConvertFrom-Json
$viewerScript = Join-Path $PSScriptRoot 'view-room.ps1'
if ($Background) {
    $launchDir = Join-Path $repoRoot "$resultDir/viewer"
    New-Item -ItemType Directory -Path $launchDir -Force | Out-Null
    $proc = Start-Process -FilePath 'powershell' -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$viewerScript`"",
            '-Checkpoint',"`"$($selected.checkpoint)`"",'-DataDir',"`"$($selected.dataset)`"",
            '-CameraIndex',$selected.camera_index,'-Port',$Port,'-OutputDir',"`"$launchDir`"") `
        -RedirectStandardOutput "$launchDir/launcher.stdout.log" -RedirectStandardError "$launchDir/launcher.stderr.log"
    @{launcher_pid=$proc.Id;port=$Port;variant=$Variant;checkpoint=$selected.checkpoint;dataset=$selected.dataset} |
        ConvertTo-Json | Set-Content -LiteralPath "$launchDir/process.json" -Encoding UTF8
    Write-Host "Small scene: http://localhost:$Port (launcher PID $($proc.Id))"
} else {
    & $viewerScript -Checkpoint $selected.checkpoint -DataDir $selected.dataset -CameraIndex $selected.camera_index -Port $Port -OutputDir "$resultDir/viewer"
}
