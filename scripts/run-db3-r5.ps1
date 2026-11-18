param(
    [ValidateSet('context','matte')][string]$Model='context',
    [ValidateRange(0,65535)][int]$Port=0,
    [switch]$Background,
    [switch]$CheckOnly,
    [string]$DeliveryDir='results/db3_r4_20261111_v2/delivery'
)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($DeliveryDir)) {$DeliveryDir=Join-Path $repoRoot $DeliveryDir}
$manifest=Get-Content -LiteralPath (Join-Path $DeliveryDir 'manifest.json') -Raw | ConvertFrom-Json
$seal=Get-Content -LiteralPath (Join-Path $DeliveryDir 'artifact_hashes.json') -Raw | ConvertFrom-Json
$configName=if ($Model -eq 'context') {'baseline_viewer.json'} else {'matte_viewer.json'}
$configPath=Join-Path $DeliveryDir $configName
foreach ($name in @('manifest.json',$configName)) {
    $expected=$seal.PSObject.Properties[$name].Value
    if (-not $expected -or (Get-FileHash -LiteralPath (Join-Path $DeliveryDir $name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw "Delivery file does not match its seal: $name"
    }
}
$config=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$selected=if ($Model -eq 'context') {$manifest.whole_route} else {$manifest.matte}
$checkpoint=Join-Path $repoRoot $config.checkpoint
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $selected.checkpoint_sha256) {
    throw 'Checkpoint does not match the selected delivery.'
}
if ($Port -eq 0) {$Port=if ($Model -eq 'context') {8094} else {8095}}
Write-Host 'R5 operational verification is available. Final reconstruction acceptance remains open.'
Write-Host "Selected model: $Model. Context and matte retain separate coordinate systems."
Write-Host "Verified checkpoint: $checkpoint"
if ($CheckOnly) {return}
& "$PSScriptRoot/view-room.ps1" -Checkpoint $config.checkpoint -DataDir $config.dataset `
    -CameraIndex $config.camera_index -Port $Port -Background:$Background `
    -OutputDir "results/db3_r5_operational_20261112/viewer-$Model"
