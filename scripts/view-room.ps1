param(
    [string]$Checkpoint = 'results/room_gtx1660/ckpts/ckpt_599_rank0.pt',
    [string]$DataDir = 'data/room_small',
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [switch]$Background,
    [int]$CameraIndex = 0,
    [string]$OutputDir = 'results/room_gtx1660/viewer'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$toolchain = Join-Path $repoRoot '.tools/msvc/setup_x64.bat'
$cudaRoot = Join-Path $repoRoot '.tools/cuda-12.8'
if (-not [IO.Path]::IsPathRooted($Checkpoint)) {
    $Checkpoint = Join-Path $repoRoot $Checkpoint
}
if (-not [IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $repoRoot $DataDir
}
foreach ($required in @($python, $toolchain, "$cudaRoot/bin/nvcc.exe", $Checkpoint, $DataDir)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing prerequisite: $required" }
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. If the room viewer is running, open http://localhost:$Port; otherwise select another -Port."
}

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
$env:NUM_CHANNELS = '3,4'
$env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
$env:PYTHONUNBUFFERED = '1'

Write-Host "Room viewer: http://localhost:$Port"
if ($Background) {
    $viewerProcess = Start-Process -FilePath $python -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -ArgumentList @('-u', 'examples/simple_viewer.py', '--ckpt', "`"$Checkpoint`"", '--camera_data_dir', "`"$DataDir`"", '--camera_index', $CameraIndex, '--port', $Port, '--output_dir', "`"$OutputDir`"") `
        -RedirectStandardOutput (Join-Path $repoRoot '.cache/room-viewer.stdout.log') `
        -RedirectStandardError (Join-Path $repoRoot '.cache/room-viewer.stderr.log')
    Write-Host "Background process ID: $($viewerProcess.Id). Stop with: taskkill.exe /PID $($viewerProcess.Id) /T /F"
} else {
    Push-Location $repoRoot
    try {
        & $python -u examples/simple_viewer.py --ckpt $Checkpoint --camera_data_dir $DataDir --camera_index $CameraIndex --port $Port --output_dir $OutputDir
        if ($LASTEXITCODE -ne 0) { throw "Viewer exited with code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }
}
