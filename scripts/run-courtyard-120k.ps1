$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$result = Join-Path $repo 'results/realsense_20261030_111354_all30_120k'
$data = Join-Path $repo 'data/realsense_20261030_111354_all30'
New-Item -ItemType Directory -Force -Path $result | Out-Null
$env:OMP_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '8'
$env:PYTHONPATH = $repo
$env:TORCH_HOME = Join-Path $repo '.cache/torch'
$env:TORCH_EXTENSIONS_DIR = Join-Path $repo '.cache/torch_extensions'
$env:TORCH_CUDA_ARCH_LIST = '7.5'
$env:NUM_CHANNELS = '3,4'
$env:MAX_JOBS = '2'
$status = Join-Path $result 'status.json'
function Write-Status($stage, $state, $detail) {
    @{stage=$stage; status=$state; detail=$detail; timestamp=(Get-Date).ToString('o'); pid=$PID} | ConvertTo-Json | Set-Content -Encoding UTF8 $status
}
try {
    Write-Status 'train' 'running' 'Fresh 120000-step run'
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test-small-dataset.ps1 `
        -Steps 120000 -DataDir (Join-Path $data 'rgbd/dataset') -ResultDir (Join-Path $result 'train') `
        -ShDegree 3 -ShDegreeInterval 1000 -RefineStart 500 -RefineStop 6000 -ResetEvery 3000 `
        -Strategy default -RecordExperiment -DataWorkers 0 -NoCacheImages `
        -SensorDepthDir (Join-Path $data 'rgbd/dataset/sensor_depth') -DepthWeight 0.01 *> (Join-Path $result 'train.log')
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
    Write-Status 'finalize' 'running' 'Evaluating matched heldout views and exporting preview'
    $ErrorActionPreference = 'Continue'
    & (Join-Path $repo '.venv/Scripts/python.exe') -u scripts/run-realsense-all30.py --stage finalize --data $data --result $result --steps 120000 *> (Join-Path $result 'finalize.log')
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "Finalization exited with code $LASTEXITCODE" }
    Write-Status 'complete' 'complete' 'Training and finalization finished'
} catch {
    Write-Status 'error' 'failed' $_.ToString()
    throw
}
