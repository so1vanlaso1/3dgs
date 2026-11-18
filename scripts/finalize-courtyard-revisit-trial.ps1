param([Parameter(Mandatory=$true)][int]$TrainingPid, [switch]$PreferPeople)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$result = Join-Path $repo $(if ($PreferPeople) { 'results/courtyard_revisit_repair_20261102/trial_people' } else { 'results/courtyard_revisit_repair_20261102/trial' })
$statusPath = Join-Path $result 'comparison_status.json'
function Save-Status($state, $detail) {
    @{status=$state; detail=$detail; timestamp=(Get-Date).ToString('o'); pid=$PID} |
        ConvertTo-Json | Set-Content -Encoding UTF8 $statusPath
}
try {
    Save-Status 'waiting' 'Waiting for trial training to finish'
    while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
    $training = Get-Content (Join-Path $result 'status.json') -Raw | ConvertFrom-Json
    if ($training.status -ne 'complete') { throw "Trial training ended with status $($training.status)" }
    $toolchainEnv = & cmd.exe /d /c "call `"$repo\.tools\msvc\setup_x64.bat`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
    foreach ($line in $toolchainEnv) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    $env:CUDA_HOME = Join-Path $repo '.tools/cuda-12.8'
    $env:CUDA_PATH = $env:CUDA_HOME
    $env:PATH = "$env:CUDA_HOME/bin;$(Join-Path $repo '.venv/Scripts');$env:PATH"
    $env:PYTHONPATH = $repo
    $env:TORCH_EXTENSIONS_DIR = Join-Path $repo '.cache/torch_extensions'
    $env:TORCH_HOME = Join-Path $repo '.cache/torch'
    $env:TORCH_CUDA_ARCH_LIST = '7.5'
    $env:NUM_CHANNELS = '3,4'
    $env:MAX_JOBS = '2'
    Save-Status 'running' 'Evaluating same 88 heldout RGB targets as original 20k model'
    $ErrorActionPreference = 'Continue'
    $compareArgs = @()
    if ($PreferPeople) { $compareArgs += '--prefer-people' }
    & (Join-Path $repo '.venv/Scripts/python.exe') -u scripts/compare-courtyard-revisit.py @compareArgs *> (Join-Path $result 'comparison.log')
    $exit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exit -ne 0) { throw "Comparison exited with code $exit" }
    Save-Status 'complete' 'Matched comparison complete; inspect geometry and results before accepting model'
} catch {
    Save-Status 'failed' $_.ToString()
    throw
}
