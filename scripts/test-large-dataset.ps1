param(
    [ValidateRange(1000, 30000)][int]$Steps = 7000,
    [string]$DataDir = 'data/room_factor2',
    [string]$ResultDir = 'results/room_hq_gtx1660',
    [string]$Checkpoint
)

# Share the verified Windows toolchain setup and trainer with the smoke test.
# Keep full prepared image resolution, batch size one, and packed rendering.
& "$PSScriptRoot/test-small-dataset.ps1" -Steps $Steps -DataDir $DataDir `
    -ResultDir $ResultDir -Checkpoint $Checkpoint -ShDegree 3 `
    -ShDegreeInterval 1000 -RefineStart 500 `
    -RefineStop ([Math]::Min(15000, [Math]::Max(600, $Steps - 1000))) -ResetEvery 3000
