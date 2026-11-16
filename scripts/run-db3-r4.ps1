param(
    [ValidateSet('screen','package','seal','view-context','view-matte')][string]$Action='view-context',
    [string]$RunDir='results/db3_r4_20261111_v2',
    [ValidateRange(1,65535)][int]$Port=8094,
    [switch]$Background
)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$python=Join-Path $repoRoot '.venv/Scripts/python.exe'
Push-Location $repoRoot
try {
    if ($Action -like 'view-*') {
        $configName=if ($Action -eq 'view-matte') {'matte_viewer.json'} else {'baseline_viewer.json'}
        $selected=Get-Content -LiteralPath (Join-Path $RunDir "delivery/$configName") -Raw | ConvertFrom-Json
        Write-Host $selected.scope
        & "$PSScriptRoot/view-room.ps1" -Checkpoint $selected.checkpoint -DataDir $selected.dataset `
            -CameraIndex $selected.camera_index -Port $Port -Background:$Background -OutputDir "$RunDir/viewer-$Action"
        return
    }
    if ($Action -eq 'screen') {
        & $python -u scripts/db3_r4_route.py --output-dir $RunDir
    } else {
        $toolchain=Join-Path $repoRoot '.tools/msvc/setup_x64.bat'
        $toolchainEnv=& cmd.exe /d /c "call `"$toolchain`" >nul && set"
        if ($LASTEXITCODE -ne 0) {throw 'MSVC environment setup failed'}
        foreach ($line in $toolchainEnv) {
            if ($line -match '^([^=]+)=(.*)$') {[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
        }
        $cudaRoot=Join-Path $repoRoot '.tools/cuda-12.8'
        $env:CUDA_HOME=$cudaRoot;$env:CUDA_PATH=$cudaRoot
        $env:PATH="$(Split-Path -Parent $python);$cudaRoot/bin;$env:PATH"
        $env:TORCH_EXTENSIONS_DIR=Join-Path $repoRoot '.cache/torch_extensions'
        $env:TORCH_CUDA_ARCH_LIST='7.5';$env:NUM_CHANNELS='3,4';$env:MAX_JOBS='2'
        $arguments=@('-u','scripts/package_db3_r4.py','--output-dir',$RunDir)
        if ($Action -eq 'seal') {$arguments+='--seal-existing'}
        & $python @arguments
    }
    if ($LASTEXITCODE -ne 0) {throw "R4 $Action exited with code $LASTEXITCODE"}
} finally {Pop-Location}
