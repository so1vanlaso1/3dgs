param(
    [ValidateRange(1000, 30000)][int]$Steps = 7000,
    [ValidateRange(112627, 10000000)][int]$GaussianBudget = 700000,
    [double]$ErrorWeight = 1.0,
    [double]$EdgeWeight = 0.0,
    [ValidateSet('random', 'long_axis')][string]$SplitMode = 'random',
    [string]$ResultDir = 'results/room_budget700k_error'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$resultPath = if ([System.IO.Path]::IsPathRooted($ResultDir)) {
    $ResultDir
} else {
    Join-Path $repoRoot $ResultDir
}
if (Test-Path -LiteralPath $resultPath) {
    throw "Result directory already exists; choose a new -ResultDir: $resultPath"
}
& "$PSScriptRoot/test-small-dataset.ps1" -Steps $Steps -DataDir 'data/room_factor2' `
    -ResultDir $resultPath -ShDegree 3 -ShDegreeInterval 1000 -RefineStart 500 `
    -RefineStop ([Math]::Min(15000, [Math]::Max(600, $Steps - 1000))) -ResetEvery 3000 `
    -Strategy budget -GaussianBudget $GaussianBudget -ErrorWeight $ErrorWeight -EdgeWeight $EdgeWeight `
    -SplitMode $SplitMode
