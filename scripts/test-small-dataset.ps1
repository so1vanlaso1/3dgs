param(
    [int]$Steps = 600,
    [string]$DataDir = 'data/smoke_colmap',
    [string]$ResultDir = 'results/smoke_gtx1660',
    [string]$Checkpoint,
    [ValidateRange(0, 3)][int]$ShDegree = 1,
    [ValidateRange(1, 30000)][int]$ShDegreeInterval = 200,
    [int]$RefineStart = 100,
    [int]$RefineStop = 500,
    [ValidateRange(1, 30000)][int]$ResetEvery = 1000,
    [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'
if ($Steps -lt 1) { throw 'Steps must be positive.' }
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$toolchain = Join-Path $repoRoot '.tools/msvc/setup_x64.bat'
$cudaRoot = Join-Path $repoRoot '.tools/cuda-12.8'
foreach ($required in @($python, $toolchain, "$cudaRoot/bin/nvcc.exe")) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing prerequisite: $required" }
}

# Import the portable MSVC environment into this PowerShell process.
$toolchainEnv = & cmd.exe /d /c "call `"$toolchain`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed.' }
foreach ($line in $toolchainEnv) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot/bin;$(Split-Path -Parent $python);$env:PATH"
$env:TORCH_EXTENSIONS_DIR = Join-Path $repoRoot '.cache/torch_extensions'
$env:TORCH_HOME = Join-Path $repoRoot '.cache/torch'
$env:TORCH_CUDA_ARCH_LIST = '7.5'
$env:MAX_JOBS = '2'
# RGB and RGB+depth are sufficient for this small 3DGS run and reduce build time.
$env:NUM_CHANNELS = '3,4'
$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
$env:PYTHONUNBUFFERED = '1'

Push-Location $repoRoot
try {
    # Windows PowerShell represents native stderr as non-terminating errors.
    # Check the process exit code after capturing its complete output.
    $ErrorActionPreference = 'Continue'
    if ($BuildOnly) {
        & $python -u -c "from gsplat.cuda._backend import _C; assert _C is not None; print('CUDA_BACKEND_READY')" 2>&1 | ForEach-Object { $_.ToString() }
    } else {
        if (-not (Test-Path -LiteralPath $DataDir)) { throw "Dataset not found: $DataDir" }
        $checkpointArgs = @()
        if ($Checkpoint) { $checkpointArgs = @('--ckpt', $Checkpoint) }
        & $python -u examples/simple_trainer.py default `
            --data-dir $DataDir --data-factor 1 --result-dir $ResultDir `
            --max-steps $Steps --eval-steps 1 $Steps --save-steps $Steps `
            --save-ply --ply-steps $Steps --disable-viewer --disable-video `
            --batch-size 1 --packed --sh-degree $ShDegree --sh-degree-interval $ShDegreeInterval --cache-images `
            --strategy.refine-start-iter $RefineStart --strategy.refine-stop-iter $RefineStop `
            --strategy.refine-every 100 --strategy.reset-every $ResetEvery @checkpointArgs 2>&1 | ForEach-Object { $_.ToString() }
    }
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($nativeExitCode -ne 0) { throw "Pipeline exited with code $nativeExitCode." }
} finally {
    Pop-Location
}
