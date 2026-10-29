param(
    [Parameter(Mandatory = $true)][int]$CampaignProcessId,
    [string]$OutputRoot = 'results/research_completion_20261027'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$process = Get-Process -Id $CampaignProcessId -ErrorAction Stop
$startTime = $process.StartTime.ToUniversalTime().ToString('o')
$monitorPath = Join-Path $OutputRoot 'continuation_monitor.json'

while ($true) {
    $process = Get-Process -Id $CampaignProcessId -ErrorAction SilentlyContinue
    $running = $null -ne $process -and $process.StartTime.ToUniversalTime().ToString('o') -eq $startTime
    & $python scripts/research_progress.py --root $OutputRoot
    $state = Get-Content -LiteralPath (Join-Path $OutputRoot 'status.json') -Raw | ConvertFrom-Json
    [ordered]@{
        campaign_process_id = $CampaignProcessId
        campaign_process_start_utc = $startTime
        process_running = $running
        campaign_status = $state.status
        checked_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $monitorPath -Encoding UTF8
    if (-not $running) {
        if ($state.status -ne 'complete' -and $state.status -ne 'failed') {
            Add-Content -LiteralPath (Join-Path $OutputRoot 'PROGRESS.md') -Encoding UTF8 -Value "`nThe campaign process exited before recording completion. Inspect the continuation logs and current run before resuming."
        }
        break
    }
    Start-Sleep -Seconds 30
}
