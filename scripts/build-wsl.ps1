param(
    [string]$Distribution = "Ubuntu",
    [string]$CudaArchitectures = "native"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProject = (wsl -d $Distribution wslpath -a $project).Trim()
if (-not $wslProject) { throw "Unable to translate the project path into WSL." }

wsl -d $Distribution -- bash -lc "cd '$wslProject' && H3CSPEED_CUDA_ARCHITECTURES='$CudaArchitectures' ./scripts/build.sh"
