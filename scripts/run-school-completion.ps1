param([Parameter(ValueFromRemainingArguments=$true)][string[]]$RunArgs)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$toolchain = Join-Path $repoRoot '.tools/msvc/setup_x64.bat'
$cudaRoot = Join-Path $repoRoot '.tools/cuda-12.8'
$toolchainEnv = & cmd.exe /d /c "call `"$toolchain`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC setup failed' }
foreach ($line in $toolchainEnv) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot/bin;$(Split-Path -Parent $python);$env:PATH"
$env:TORCH_EXTENSIONS_DIR = Join-Path $repoRoot '.cache/torch_extensions'
$env:TORCH_HOME = Join-Path $repoRoot '.cache/torch'
$env:TORCH_CUDA_ARCH_LIST = '7.5'
$env:MAX_JOBS = '2'
$env:NUM_CHANNELS = '3,4'
$env:PYTHONPATH = "$repoRoot;$repoRoot/examples;$repoRoot/scripts;$env:PYTHONPATH"
$env:PYTHONUNBUFFERED = '1'
$env:TQDM_MININTERVAL = '30'
& $python @RunArgs
if ($LASTEXITCODE -ne 0) { throw "School completion command failed: $LASTEXITCODE" }
