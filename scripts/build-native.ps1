param(
    [string]$BuildDirectory = "build-native",
    [string]$CudaArchitectures = "86",
    [ValidateSet("Release", "RelWithDebInfo", "Debug")]
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

# This value is interpolated into the vcvars-backed cmd.exe command below.
# Accept only CMake's numeric, semicolon-separated CUDA architecture syntax so
# shell metacharacters can never become part of that command.
if ($CudaArchitectures -notmatch '^[0-9]+(;[0-9]+)*$') {
    throw "CudaArchitectures must be a semicolon-separated list of numeric CUDA architecture IDs."
}
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$icuRoot = Join-Path $project "third_party\icu"
$icuFiles = @(
    (Join-Path $icuRoot "include\unicode\uchar.h"),
    (Join-Path $icuRoot "lib64\icuuc.lib"),
    (Join-Path $icuRoot "bin64\icuuc76.dll"),
    (Join-Path $icuRoot "bin64\icudt76.dll")
)

if ($icuFiles.Where({ -not (Test-Path -LiteralPath $_) }).Count -ne 0) {
    if (Test-Path -LiteralPath $icuRoot) {
        throw "ICU dependency is incomplete at $icuRoot; move it aside and rerun."
    }
    $icuUrl = "https://github.com/unicode-org/icu/releases/download/release-76-1/icu4c-76_1-Win64-MSVC2022.zip"
    $icuSha256 = "BEDBA77DD1FECA09E9AE9922109A285C0ECF46D09C80B65EAE6EAE63A4E155DC"
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ("h3cspeed-icu-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temporary | Out-Null
    try {
        $archive = Join-Path $temporary "icu.zip"
        $extracted = Join-Path $temporary "extract"
        Invoke-WebRequest -Uri $icuUrl -OutFile $archive
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
        if ($actual -ne $icuSha256) {
            throw "ICU archive SHA-256 mismatch: expected $icuSha256, got $actual"
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $extracted
        New-Item -ItemType Directory -Path (Split-Path $icuRoot) -Force | Out-Null
        Move-Item -LiteralPath $extracted -Destination $icuRoot
    }
    finally {
        $tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        $resolved = [IO.Path]::GetFullPath($temporary)
        if ($resolved.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase) -and
            (Test-Path -LiteralPath $resolved)) {
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}

python (Join-Path $project "scripts\bootstrap.py") --project-root $project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) { throw "vswhere.exe was not found." }
$visualStudio = (& $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath).Trim()
if (-not $visualStudio) { throw "Visual Studio C++ Build Tools were not found." }
$vcvars = Join-Path $visualStudio "VC\Auxiliary\Build\vcvars64.bat"
$build = [IO.Path]::GetFullPath((Join-Path $project $BuildDirectory))
if ($build.IndexOfAny([char[]]'&|<>^%`"') -ge 0) {
    throw "BuildDirectory contains characters that are unsafe for cmd.exe."
}
$command = 'call "{0}" && cmake -S "{1}" -B "{2}" -G Ninja -DCMAKE_BUILD_TYPE={3} -DH3CSPEED_CUDA_ARCHITECTURES={4} && cmake --build "{2}" --parallel' -f `
    $vcvars, $project, $build, $BuildType, $CudaArchitectures
cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built: $build\h3cspeed.exe"
Write-Host "Built: $build\h3cspeed-cuda-info.exe"
