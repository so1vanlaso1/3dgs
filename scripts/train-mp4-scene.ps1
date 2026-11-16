param(
    [string]$Video,
    [ValidateRange(0.1,30)][double]$Fps = 30,
    [ValidateRange(320,1920)][int]$Width = 960,
    [ValidateRange(1000,500000)][int]$Steps = 120000,
    [ValidateRange(10000,2000000)][int]$GaussianBudget = 1000000,
    [ValidateRange(1,65535)][int]$Port = 8091,
    [ValidateRange(0.1,1.0)][double]$MinRegistration = 0.8,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not $Video) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'Select the MP4 video to reconstruct'
    $dialog.Filter = 'MP4 video (*.mp4)|*.mp4|Video files (*.mov;*.mkv;*.avi)|*.mov;*.mkv;*.avi|All files (*.*)|*.*'
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { Write-Host 'No video selected.'; exit 1 }
    $Video = $dialog.FileName
}
$videoPath = (Resolve-Path -LiteralPath $Video).Path
if (-not (Test-Path -LiteralPath $videoPath -PathType Leaf)) { throw "Video not found: $Video" }
$python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project Python environment not found: .venv/Scripts/python.exe' }
& $python -c "import pycolmap, imageio_ffmpeg, torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable'"
if ($LASTEXITCODE -ne 0) { throw 'Missing Python dependencies or CUDA GPU; see error above.' }
$name = [regex]::Replace([IO.Path]::GetFileNameWithoutExtension($videoPath), '[^a-zA-Z0-9_-]+', '_').Trim('_')
if (-not $name) { $name = 'video' }
$runName = "mp4_${name}_$(Get-Date -Format 'yyyyMMdd_HHmmss')_$([guid]::NewGuid().ToString('N').Substring(0,6))"
$work = Join-Path $repo "data/$runName"
$result = Join-Path $repo "results/$runName"
New-Item -ItemType Directory -Path $work, $result | Out-Null
$transcript = Join-Path $result 'pipeline.log'
Start-Transcript -Path $transcript | Out-Null
try {
    Write-Host "Video: $videoPath"
    Write-Host "Working data: $work"
    Write-Host "Results: $result"
    $invariant = [Globalization.CultureInfo]::InvariantCulture
    $ErrorActionPreference = 'Continue'
    & $python -u (Join-Path $PSScriptRoot 'mp4-to-3dgs-prepare.py') --video $videoPath --work $work --fps $Fps.ToString($invariant) --width $Width --min-registration $MinRegistration.ToString($invariant) 2>&1 | ForEach-Object { $_.ToString() }
    $prepareExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($prepareExit -ne 0) { throw "Reconstruction failed; see $work/status.json and $work/ffmpeg.log." }
    if ($PrepareOnly) { Write-Host "Preparation complete. Dataset: $work/dataset"; exit 0 }
    Write-Host 'STAGE: Train RGB-only Gaussian scene (this may take a long time)'
    $stop = [Math]::Min(6000, [Math]::Max(600, $Steps - 1000))
    & (Join-Path $PSScriptRoot 'test-small-dataset.ps1') -Steps $Steps -DataDir "$work/dataset" -ResultDir $result -Strategy budget -GaussianBudget $GaussianBudget -NoCacheImages -ShDegree 3 -ShDegreeInterval 1000 -RefineStart 500 -RefineStop $stop -ResetEvery 3000
    $ckpt = Get-ChildItem -LiteralPath (Join-Path $result 'ckpts') -Filter 'ckpt_*_rank0.pt' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $ckpt) { throw 'Training finished without a checkpoint; inspect pipeline.log.' }
    $viewer = Join-Path $PSScriptRoot 'view-room.ps1'
    $viewerCommand = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$viewer`" -Checkpoint `"$($ckpt.FullName)`" -DataDir `"$work/dataset`" -Port $Port -OutputDir `"$result/viewer`" -Background"
    $summary = @{video=$videoPath;dataset="$work/dataset";checkpoint=$ckpt.FullName;result=$result;port=$Port;viewer_command=$viewerCommand;depth_used=$false;registration_gate='80 percent by default; not proof of coverage'}
    $summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $result 'scene.json') -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $result 'start-viewer.ps1') -Value $viewerCommand -Encoding UTF8
    Write-Host "Training complete. Checkpoint: $($ckpt.FullName)"
    Write-Host "Viewer restart script: $result/start-viewer.ps1"
    try {
        & $viewer -Checkpoint $ckpt.FullName -DataDir "$work/dataset" -Port $Port -OutputDir "$result/viewer" -Background
        $viewerReady = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            try { $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$Port" -TimeoutSec 2; if ($response.StatusCode -eq 200) { $viewerReady = $true; break } } catch {}
            Start-Sleep -Seconds 1
        }
        if ($viewerReady) { Write-Host "Open http://localhost:$Port" } else { Write-Warning "Viewer did not respond yet. Check .cache/room-viewer.stderr.log." }
    } catch { Write-Warning "Viewer did not start: $_. Training output is saved. Restart using $result/start-viewer.ps1 or select another port." }
} finally { Stop-Transcript | Out-Null }
exit 0
