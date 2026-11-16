param([int]$Port=8087, [switch]$Background, [switch]$PreferPeople)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
Set-Location $repo
$trial=if ($PreferPeople) { 'trial_people' } else { 'trial' }
$dataset=if ($PreferPeople) { 'data/courtyard_revisit_people_20261102' } else { 'data/courtyard_revisit_repair_20261102' }
$checkpoint=Join-Path $repo "results/courtyard_revisit_repair_20261102/$trial/train/ckpts/final_post_update.pt"
$data=Join-Path $repo $dataset
$python=Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $checkpoint)) { throw 'Candidate checkpoint not yet available; check trial/status.json.' }
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
$envLines=& cmd.exe /d /c "call `"$repo\.tools\msvc\setup_x64.bat`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC setup failed' }
foreach ($line in $envLines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:CUDA_HOME=Join-Path $repo '.tools/cuda-12.8'
$env:CUDA_PATH=$env:CUDA_HOME
$env:PATH="$env:CUDA_HOME/bin;$(Join-Path $repo '.venv/Scripts');$env:PATH"
$env:PYTHONPATH=$repo
$env:TORCH_EXTENSIONS_DIR=Join-Path $repo '.cache/torch_extensions'
$env:TORCH_HOME=Join-Path $repo '.cache/torch'
$env:TORCH_CUDA_ARCH_LIST='7.5'
$env:NUM_CHANNELS='3,4'
$env:MAX_JOBS='2'
$out=Join-Path $repo "results/courtyard_revisit_repair_20261102/$trial/viewer"
$arguments=@('-u','examples/simple_viewer.py','--ckpt',"`"$checkpoint`"",'--camera_data_dir',"`"$data`"",'--camera_index','3206','--port',$Port,'--output_dir',"`"$out`"")
Write-Host "Candidate viewer: http://localhost:$Port"
if ($Background) {
    $p=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $repo '.cache/courtyard-revisit-viewer.stdout.log') `
        -RedirectStandardError (Join-Path $repo '.cache/courtyard-revisit-viewer.stderr.log')
    Write-Host "Background process ID: $($p.Id)"
} else { & $python @arguments }
