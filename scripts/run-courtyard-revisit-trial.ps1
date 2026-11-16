param([switch]$PreferPeople)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$result = Join-Path $repo $(if ($PreferPeople) { 'results/courtyard_revisit_repair_20261102/trial_people' } else { 'results/courtyard_revisit_repair_20261102/trial' })
$data = Join-Path $repo $(if ($PreferPeople) { 'data/courtyard_revisit_people_20261102' } else { 'data/courtyard_revisit_repair_20261102' })
New-Item -ItemType Directory -Force -Path $result | Out-Null
function Save-Status($state, $detail) {
    @{status=$state; detail=$detail; timestamp=(Get-Date).ToString('o'); pid=$PID} |
        ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $result 'status.json')
}
try {
    $description = if ($PreferPeople) { 'Experimental corrected pose; 20000 updates; late revisit training frames excluded to retain people' } else { 'Experimental corrected-pose scene; 20000 updates; moving people unmasked' }
    Save-Status 'running' $description
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test-small-dataset.ps1 `
        -Steps 20000 -DataDir $data -ResultDir (Join-Path $result 'train') `
        -ShDegree 3 -ShDegreeInterval 1000 -RefineStart 500 -RefineStop 6000 -ResetEvery 3000 `
        -Strategy default -RecordExperiment -DataWorkers 0 -NoCacheImages `
        -SensorDepthDir (Join-Path $data 'sensor_depth') -DepthWeight 0.01 `
        *> (Join-Path $result 'train.log')
    $exit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($exit -ne 0) { throw "Training exited with code $exit" }
    Save-Status 'complete' 'Training finished; independent matched comparison remains required'
} catch {
    Save-Status 'failed' $_.ToString()
    throw
}
