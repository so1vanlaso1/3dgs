param(
    [ValidateRange(1,65535)][int]$Port = 8084,
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot/view-clean-realsense.ps1" -ResultDir 'results/realsense_20261030_111354_all30' `
    -Port $Port -Background:$Background
