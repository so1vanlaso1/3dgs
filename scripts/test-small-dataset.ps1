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
    [switch]$BuildOnly,
    [ValidateSet('default', 'budget')][string]$Strategy = 'default',
    [ValidateRange(1, 10000000)][int]$GaussianBudget = 700000,
    [double]$ErrorWeight = 1.0,
    [double]$EdgeWeight = 0.0,
    [ValidateSet('random', 'long_axis')][string]$SplitMode = 'random',
    [string]$SensorDepthDir,
    [double]$DepthWeight = 0.01,
    [int]$TestEvery = 8,
    [switch]$RecordExperiment,
    [int]$DataWorkers = 4,
    [switch]$NoCacheImages,
    [int[]]$Milestones = @(),
    [ValidateSet('method_default', 'clone', 'split')][string]$ClonePolicy = 'method_default'
)

$ErrorActionPreference = 'Stop'
# Optional run-specific override, read at training launch after preparation.
# Only an explicit sidecar next to this dataset can change these settings.
$datasetPath = [IO.Path]::GetFullPath($DataDir)
$overridePath = Join-Path (Split-Path -Parent $datasetPath) 'training_override.json'
if (Test-Path -LiteralPath $overridePath -PathType Leaf) {
    $override = Get-Content -LiteralPath $overridePath -Raw | ConvertFrom-Json
    if ($override.schema -ne 'gsplat_training_override_v1' -or
        $override.steps -lt 1000 -or $override.steps -gt 500000 -or
        $override.gaussian_budget -lt 10000 -or $override.gaussian_budget -gt 10000000) {
        throw "Invalid training override: $overridePath"
    }
    $Steps = [int]$override.steps
    $GaussianBudget = [int]$override.gaussian_budget
    Write-Host "Run-specific override from $overridePath : steps=$Steps GaussianBudget=$GaussianBudget"
}
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
        $strategyArgs = @()
        $experimentArgs = @('--test-every', $TestEvery, '--data-workers', $DataWorkers)
        if (-not $NoCacheImages) { $experimentArgs += '--cache-images' }
        if ($SensorDepthDir) {
            $experimentArgs += @('--depth-loss', '--sensor-depth-dir', $SensorDepthDir,
                '--depth-lambda', $DepthWeight)
        }
        if ($RecordExperiment) {
            $experimentArgs += @('--record-experiment', '--independent-sampling')
        }
        $evalMilestones = @(1) + $Milestones + @($Steps) | Select-Object -Unique
        $saveMilestones = $Milestones + @($Steps) | Select-Object -Unique
        if ($Strategy -eq 'budget') {
            $strategyArgs = @('--strategy.cap-max', $GaussianBudget,
                '--strategy.error-weight', $ErrorWeight, '--strategy.edge-weight', $EdgeWeight,
                '--strategy.split-mode', $SplitMode, '--strategy.clone-policy', $ClonePolicy)
        }
        & $python -u examples/simple_trainer.py $Strategy `
            --data-dir $DataDir --data-factor 1 --result-dir $ResultDir `
            --max-steps $Steps --eval-steps @evalMilestones --save-steps @saveMilestones `
            --save-ply --ply-steps $Steps --disable-viewer --disable-video `
            --batch-size 1 --packed --sh-degree $ShDegree --sh-degree-interval $ShDegreeInterval `
            --strategy.refine-start-iter $RefineStart --strategy.refine-stop-iter $RefineStop `
            --strategy.refine-every 100 --strategy.reset-every $ResetEvery @strategyArgs @experimentArgs @checkpointArgs 2>&1 | ForEach-Object { $_.ToString() }
    }
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($nativeExitCode -ne 0) { throw "Pipeline exited with code $nativeExitCode." }
} finally {
    Pop-Location
}
