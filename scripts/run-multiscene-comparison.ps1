param(
    [string[]]$Scenes = @('counter', 'kitchen', 'bonsai'),
    [int]$Steps = 7000,
    [string]$OutputRoot = 'results/multiscene_a04_20261022'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
foreach ($scene in $Scenes) {
    if ($scene -notin @('counter', 'kitchen', 'bonsai')) { throw "Unsupported campaign scene: $scene" }
    $dataDir = "data/${scene}_factor2"
    if (-not (Test-Path "$dataDir/dataset_manifest.json")) { throw "Prepare $dataDir first." }
    $sceneRoot = Join-Path $OutputRoot $scene
    New-Item -ItemType Directory -Force -Path $sceneRoot | Out-Null
    foreach ($method in @('B00', 'A04')) {
        $runDir = Join-Path $sceneRoot $method
        $finalStats = Join-Path $runDir ('stats/val_step{0:d4}.json' -f ($Steps - 1))
        if (-not (Test-Path $finalStats)) {
            if (Test-Path $runDir) { throw "Incomplete run exists; inspect before retry: $runDir" }
            Write-Output "START $scene $method $(Get-Date -Format o)"
            $runArgs = @{
                Steps = $Steps; DataDir = $dataDir; ResultDir = $runDir
                ShDegree = 3; ShDegreeInterval = 1000
                RefineStart = 500; RefineStop = 6000; ResetEvery = 3000
            }
            if ($method -eq 'A04') {
                $runArgs.Strategy = 'budget'
                $runArgs.GaussianBudget = 700000
                $runArgs.ErrorWeight = 1.0
                $runArgs.EdgeWeight = 0.25
                $runArgs.SplitMode = 'long_axis'
            }
            & "$PSScriptRoot/test-small-dataset.ps1" @runArgs |
                Out-File -FilePath (Join-Path $sceneRoot "$method.log") -Encoding utf8
            if (-not (Test-Path $finalStats)) { throw "Missing completed evaluation: $scene $method" }
            Write-Output "DONE $scene $method $(Get-Content $finalStats -Raw)"
        }
    }
    $reloadDir = Join-Path $sceneRoot 'A04_reload'
    $reloadStats = Join-Path $reloadDir ('stats/val_step{0:d4}.json' -f ($Steps - 1))
    if (-not (Test-Path $reloadStats)) {
        if (Test-Path $reloadDir) { throw "Incomplete reload exists: $reloadDir" }
        $ckpt = Join-Path $sceneRoot ('A04/ckpts/ckpt_{0}_rank0.pt' -f ($Steps - 1))
        & "$PSScriptRoot/test-small-dataset.ps1" -Steps $Steps -DataDir $dataDir `
            -ResultDir $reloadDir -Checkpoint $ckpt -ShDegree 3 -ShDegreeInterval 1000 `
            -RefineStart 500 -RefineStop 6000 -ResetEvery 3000 -Strategy budget `
            -GaussianBudget 700000 -ErrorWeight 1 -EdgeWeight 0.25 -SplitMode long_axis |
            Out-File -FilePath (Join-Path $sceneRoot 'A04_reload.log') -Encoding utf8
    }
    & $python scripts/report-budget-experiment.py --baseline "$sceneRoot/B00" `
        --run "A04=$sceneRoot/A04" --reload "A04=$reloadDir" --output "$sceneRoot/comparison.json"
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $scene" }
    Write-Output "VERIFIED $scene"
}
Write-Output 'CAMPAIGN_COMPLETE'
