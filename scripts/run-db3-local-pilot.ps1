param([ValidateSet('matte','exhibit')][string]$Region='matte', [string[]]$Arms=@('off','current','confidence'), [ValidateSet('train','evaluate')][string]$Action='train', [int]$Steps=7000, [string]$CandidateDir='')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$python=Join-Path $repoRoot '.venv/Scripts/python.exe'
$toolchain=Join-Path $repoRoot '.tools/msvc/setup_x64.bat'
$toolchainEnv=& cmd.exe /d /c "call `"$toolchain`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC setup failed' }
foreach ($line in $toolchainEnv) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process') }
}
$cudaRoot=Join-Path $repoRoot '.tools/cuda-12.8'
$env:CUDA_HOME=$cudaRoot
$env:CUDA_PATH=$cudaRoot
$env:PATH="$cudaRoot/bin;$(Split-Path -Parent $python);$env:PATH"
$env:TORCH_EXTENSIONS_DIR=Join-Path $repoRoot '.cache/torch_extensions'
$env:TORCH_HOME=Join-Path $repoRoot '.cache/torch'
$env:TORCH_CUDA_ARCH_LIST='7.5'
$env:MAX_JOBS='2'
$env:NUM_CHANNELS='3,4'
$env:PYTHONUNBUFFERED='1'
$env:TQDM_MININTERVAL='30'
if ($Action -eq 'train') {
    & $python -u scripts/db3_local_pilot.py train --region $Region --steps $Steps --arms @Arms
} else {
    $evalArgs=@('scripts/db3_evaluate_pilot.py','evaluate','--region',$Region)
    if ($CandidateDir) { $evalArgs+=@('--candidate-dir',$CandidateDir,'--steps',$Steps) }
    & $python -u @evalArgs
}
if ($LASTEXITCODE -ne 0) { throw "Local pilot exited with code $LASTEXITCODE" }
