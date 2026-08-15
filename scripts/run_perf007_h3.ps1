[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ModelRoot,
    [Parameter(Mandatory = $true)] [string]$ComfyUIRoot,
    [Parameter(Mandatory = $true)] [string]$TextEncoder,
    [Parameter(Mandatory = $true)] [string]$PromptFile,
    [Parameter(Mandatory = $true)] [string]$FirstFrame,
    [Parameter(Mandatory = $true)] [string]$Output,
    [Parameter(Mandatory = $true)] [string]$SidecarPath,
    [Parameter(Mandatory = $true)] [string]$BinaryPath,
    [Parameter(Mandatory = $true)] [string]$ProfileDir,
    [int]$Width = 864,
    [int]$Height = 480,
    [int]$RenderWidth = 0,
    [int]$RenderHeight = 0,
    [int]$Frames = 124,
    [int]$Steps = 2,
    [int]$Layers = 50,
    [int]$Reuse = 1,
    [int]$CoreReuse = 1,
    [int]$WeightCacheMib = 1536,
    [ValidateRange(1, 12)]
    [int]$PrefetchMaxWeights = 8,
    [UInt64]$Seed = 42,
    [switch]$LayerMajor,
    [switch]$AsyncRefill,
    [switch]$DitPrefetch,
    [switch]$ResolutionMatrix,
    [switch]$UseExistingSidecar
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-File([string]$Value, [string]$Label) {
    $path = [IO.Path]::GetFullPath($Value)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "$Label does not exist: $path"
    }
    return $path
}

function Resolve-Directory([string]$Value, [string]$Label) {
    $path = [IO.Path]::GetFullPath($Value)
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "$Label does not exist: $path"
    }
    return $path
}

function Get-Sha256([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $stream = [IO.File]::OpenRead($Path)
        try {
            return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha.Dispose()
    }
}

function Get-PngDimensions([string]$Path) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 24 -or
        $bytes[0] -ne 0x89 -or $bytes[1] -ne 0x50 -or
        $bytes[2] -ne 0x4e -or $bytes[3] -ne 0x47 -or
        $bytes[4] -ne 0x0d -or $bytes[5] -ne 0x0a -or
        $bytes[6] -ne 0x1a -or $bytes[7] -ne 0x0a -or
        $bytes[12] -ne 0x49 -or $bytes[13] -ne 0x48 -or
        $bytes[14] -ne 0x44 -or $bytes[15] -ne 0x52) {
        throw "first-frame must be a valid PNG"
    }
    $width = (([int]$bytes[16] -shl 24) -bor
              ([int]$bytes[17] -shl 16) -bor
              ([int]$bytes[18] -shl 8) -bor [int]$bytes[19])
    $height = (([int]$bytes[20] -shl 24) -bor
               ([int]$bytes[21] -shl 16) -bor
               ([int]$bytes[22] -shl 8) -bor [int]$bytes[23])
    if ($width -le 0 -or $height -le 0) { throw "first-frame PNG has invalid dimensions" }
    return @($width, $height)
}

if (-not $ResolutionMatrix -and ($Width -ne 864 -or $Height -ne 480)) {
    throw "PERF-007 requires the native 864x480 480p contract"
}
if ($ResolutionMatrix -and ($Width -lt 64 -or $Height -lt 64 -or
    ($Width % 32) -ne 0 -or ($Height % 32) -ne 0 -or
    ($Width * $Height) -gt (768 * 1344))) {
    throw "resolution matrix requires dimensions >=64, 32-pixel aligned, and within the H3 canvas limit"
}
if (($RenderWidth -eq 0) -ne ($RenderHeight -eq 0)) {
    throw "RenderWidth and RenderHeight must be set together"
}
if ($RenderWidth -eq 0) {
    $RenderWidth = $Width
    $RenderHeight = $Height
}
if ($ResolutionMatrix -and ($RenderWidth -ne $Width -or $RenderHeight -ne $Height)) {
    throw "resolution matrix forbids internal downscale or output stretching"
}
if ($RenderWidth -lt 32 -or $RenderHeight -lt 32 -or
    ($RenderWidth % 32) -ne 0 -or ($RenderHeight % 32) -ne 0 -or
    $RenderWidth -gt $Width -or $RenderHeight -gt $Height -or
    ($RenderWidth * $Height) -ne ($RenderHeight * $Width)) {
    throw "internal render size must preserve aspect ratio and use 32-pixel multiples"
}
if ($Frames -lt 22 -or (($Frames - 5) % 17) -ne 0) {
    throw "PERF-007 frames must follow H3's 5 + 17n alignment"
}
if ($Steps -ne 2 -or $Layers -ne 50 -or $Reuse -ne 1 -or
    ($CoreReuse -ne 1 -and $CoreReuse -ne 4)) {
    throw "PERF-007 fixes steps=2, layers=50, reuse=1, core-reuse=1 or 4"
}
if ($WeightCacheMib -lt 1536 -or $WeightCacheMib -gt 4096) {
    throw "PERF-007 weight cache must be between 1536 and 4096 MiB"
}

$modelRoot = Resolve-Directory $ModelRoot "model root"
$comfyRoot = Resolve-Directory $ComfyUIRoot "ComfyUI root"
$encoder = Resolve-File $TextEncoder "text encoder"
$promptPath = Resolve-File $PromptFile "prompt file"
$firstFrame = Resolve-File $FirstFrame "first-frame image"
$binary = Resolve-File $BinaryPath "h3cspeed binary"
$output = [IO.Path]::GetFullPath($Output)
$sidecar = [IO.Path]::GetFullPath($SidecarPath)
$profile = [IO.Path]::GetFullPath($ProfileDir)
if ($ResolutionMatrix) {
    $pngDimensions = Get-PngDimensions $firstFrame
    if ($pngDimensions[0] -ne $Width -or $pngDimensions[1] -ne $Height) {
        throw "resolution matrix first-frame PNG must exactly match output dimensions"
    }
}
if (Test-Path -LiteralPath $output) { throw "output already exists: $output" }
if (-not $UseExistingSidecar -and (Test-Path -LiteralPath $sidecar)) {
    throw "sidecar already exists: $sidecar"
}
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($output)), $profile | Out-Null

$pythonCandidates = @(
    (Join-Path $comfyRoot ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $comfyRoot) ".venv\Scripts\python.exe")
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($python)) { throw "ComfyUI venv Python was not found" }
$helper = Resolve-File (Join-Path $PSScriptRoot "encode_h3_quantized_prompt.py") "conditioning helper"
$prompt = Get-Content -LiteralPath $promptPath -Raw

if ($UseExistingSidecar) {
    Write-Host "[perf007] reusing the existing bound FL2VA sidecar"
} else {
    Write-Host "[perf007] generating the bound FL2VA sidecar"
    $helperArgs = @(
        $helper, "--comfyui", $comfyRoot, "--text-encoder", $encoder,
        "--output", $sidecar, "--prompt", $prompt, "--device", "cuda:0",
        "--mode", "fl2va-i2v", "--first-frame", $firstFrame,
        "--width", $RenderWidth.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--height", $RenderHeight.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    & $python @helperArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
        throw "conditioning helper failed with exit code $LASTEXITCODE"
    }
}
$encoderHash = Get-Sha256 $encoder
$cudaEnv = @{
    H3CSPEED_TEXT_EMBEDDING = $sidecar
    H3CSPEED_TEXT_ENCODER_SHA256 = $encoderHash
    H3_FFMPEG = "C:\ffmpeg\bin\ffmpeg.exe"
    H3_CUDA_ATTENTION = "sage"
    H3_CUDA_DEVICE = "0"
    H3_CUDA_LOW_VRAM = "1"
    H3_CUDA_OFFLOAD = "ram+file"
    H3_CUDA_PINNED_HOST_MIB = "128"
    H3_CUDA_STAGING_MIB = "64"
    H3_CUDA_TF32 = "0"
    H3_CUDA_VRAM_BUDGET_MIB = "5888"
    H3_CUDA_WEIGHT_CACHE_MIB = $WeightCacheMib.ToString([Globalization.CultureInfo]::InvariantCulture)
    H3_PROFILE = "1"
    H3_PROFILE_JSON_DIR = $profile
    H3_CUDA_UPLOAD_WAIT_TRACE = "1"
    H3_VAE_LAYER_MAJOR = if ($LayerMajor) { "1" } else { "0" }
    H3_CUDA_ASYNC_REFILL = if ($AsyncRefill) { "1" } else { "0" }
    H3_CUDA_DIT_PREFETCH = if ($DitPrefetch) { "1" } else { "0" }
    H3_CUDA_DIT_PREFETCH_MAX_WEIGHTS = [string]$PrefetchMaxWeights
}
$oldEnv = @{}
try {
    foreach ($entry in $cudaEnv.GetEnumerator()) {
        $oldEnv[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
        [Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, "Process")
    }
    $firstCanonical = "$sidecar.first.png"
    if (-not (Test-Path -LiteralPath $firstCanonical -PathType Leaf)) {
        throw "conditioning helper did not publish the canonical first frame"
    }
    $cliArgs = @(
        "-d", $modelRoot, "-p", $prompt,
        "--width", $Width.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--height", $Height.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--frames", $Frames.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--steps", $Steps.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--layers", $Layers.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--reuse", $Reuse.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--core-reuse", $CoreReuse.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--seed", $Seed.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--first-frame", $firstCanonical, "-o", $output, "--profile"
    )
    if ($RenderWidth -ne $Width -or $RenderHeight -ne $Height) {
        $cliArgs = @(
            "-d", $modelRoot, "-p", $prompt,
            "--width", $Width.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--height", $Height.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--render-width", $RenderWidth.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--render-height", $RenderHeight.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--frames", $Frames.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--steps", $Steps.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--layers", $Layers.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--reuse", $Reuse.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--core-reuse", $CoreReuse.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--seed", $Seed.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--first-frame", $firstCanonical, "-o", $output, "--profile"
        )
    }
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $binary @cliArgs
    $exitCode = $LASTEXITCODE
    $watch.Stop()
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) {
        throw "h3cspeed failed with exit code $exitCode"
    }
    [pscustomobject]@{
        schema_version = 1
        kind = "h3cspeed.perf007.talking"
        frames = $Frames
        fps = 24
        width = $Width
        height = $Height
        render_width = $RenderWidth
        render_height = $RenderHeight
        resolution_matrix = [bool]$ResolutionMatrix
        steps = $Steps
        layer_major = [bool]$LayerMajor
        async_refill = [bool]$AsyncRefill
        dit_prefetch = [bool]$DitPrefetch
        prefetch_max_weights = $PrefetchMaxWeights
        weight_cache_mib = $WeightCacheMib
        wall_seconds = [math]::Round($watch.Elapsed.TotalSeconds, 6)
        media = $output
        sidecar = $sidecar
        status = "PASS"
    } | ConvertTo-Json -Depth 4
} finally {
    foreach ($entry in $oldEnv.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
