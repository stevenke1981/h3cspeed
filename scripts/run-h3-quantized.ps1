[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelRoot,

    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,

    [Parameter(Mandatory = $true)]
    [string]$TextEncoder,

    [Parameter(Mandatory = $true)]
    [string]$Prompt,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [int]$Steps = 20,
    [int]$Width = 256,
    [int]$Height = 256,
    [int]$Frames = 22,
    [int]$Layers = 50,
    [int]$Reuse = 1,
    [int]$CoreReuse = 1,
    [string]$Device = "cuda:0",
    [string]$ComfyPython = "",
    [string]$BinaryPath = "",
    [string]$SidecarPath = "",
    [string]$FirstFrame = "",
    [string]$LastFrame = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-AbsolutePath {
    param([Parameter(Mandatory = $true)][string]$Value)

    $expanded = [Environment]::ExpandEnvironmentVariables($Value)
    if ([IO.Path]::IsPathRooted($expanded)) {
        return [IO.Path]::GetFullPath($expanded)
    }
    return [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $expanded))
}

function Resolve-ExistingDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $path = ConvertTo-AbsolutePath $Value
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "$Label does not exist or is not a directory: $path"
    }
    return $path
}

function Resolve-ExistingFile {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $path = ConvertTo-AbsolutePath $Value
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "$Label does not exist or is not a file: $path"
    }
    return $path
}

function Resolve-ComfyPython {
    param(
        [Parameter(Mandatory = $true)][string]$ComfyRoot,
        [string]$ExplicitPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        return Resolve-ExistingFile $ExplicitPath "ComfyUI Python"
    }

    # The path is derived from the caller's ComfyUI root; no machine-specific
    # model or user directory is assumed.  An explicit -ComfyPython is required
    # when the checkout uses a different virtual-environment layout.
    $comfyParent = Split-Path -Parent $ComfyRoot
    $candidates = @(
        (Join-Path $ComfyRoot ".venv\Scripts\python.exe"),
        (Join-Path $ComfyRoot "venv\Scripts\python.exe"),
        (Join-Path $comfyParent ".venv\Scripts\python.exe"),
        (Join-Path $comfyParent "venv\Scripts\python.exe")
    ) | Select-Object -Unique
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (ConvertTo-AbsolutePath $candidate)
        }
    }
    throw "ComfyUI venv Python was not found below $ComfyRoot; pass -ComfyPython explicitly"
}

function Resolve-H3Binary {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [string]$ExplicitPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        return Resolve-ExistingFile $ExplicitPath "h3cspeed binary"
    }

    # Prefer the quantized build, then the normal native and portable layouts.
    # These are repository-relative candidates, not host-specific defaults.
    $candidates = @(
        (Join-Path $RepositoryRoot "bin\h3cspeed.exe"),
        (Join-Path $RepositoryRoot "build-quant\h3cspeed.exe"),
        (Join-Path $RepositoryRoot "build-native\h3cspeed.exe"),
        (Join-Path $RepositoryRoot "build\h3cspeed.exe"),
        (Join-Path $RepositoryRoot "build\h3cspeed")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (ConvertTo-AbsolutePath $candidate)
        }
    }
    throw "h3cspeed binary was not found; pass -BinaryPath explicitly or build the project"
}

function Restore-ProcessEnvironmentVariable {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowNull()][string]$Value
    )

    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

$exitCode = 2
$oldSidecar = [Environment]::GetEnvironmentVariable("H3CSPEED_TEXT_EMBEDDING", "Process")
$oldEncoderSha = [Environment]::GetEnvironmentVariable("H3CSPEED_TEXT_ENCODER_SHA256", "Process")

try {
    if ([string]::IsNullOrWhiteSpace($Prompt)) {
        throw "Prompt must not be empty"
    }
    if ($Steps -lt 1 -or $Steps -gt 100) {
        throw "Steps must be between 1 and 100"
    }
    if ($Width -lt 64 -or $Height -lt 64) {
        throw "Width and Height must be at least 64"
    }
    if ($Frames -lt 22) {
        throw "Frames must be at least the trained 22-frame decoder chunk"
    }
    if ($Layers -lt 1 -or $Layers -gt 50) {
        throw "Layers must be between 1 and 50"
    }
    if ($Reuse -lt 1 -or $CoreReuse -lt 1) {
        throw "Reuse and CoreReuse must be positive"
    }
    if ($Reuse -gt 1 -and $CoreReuse -gt 1) {
        throw "Reuse > 1 and CoreReuse > 1 cannot be combined"
    }
    if ([string]::IsNullOrWhiteSpace($Device) -or -not $Device.StartsWith("cuda", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Device must be a CUDA device (for example cuda:0); CPU fallback is forbidden"
    }

    $modelRoot = Resolve-ExistingDirectory $ModelRoot "Model root"
    $comfyRoot = Resolve-ExistingDirectory $ComfyUIRoot "ComfyUI root"
    $encoder = Resolve-ExistingFile $TextEncoder "Quantized Qwen text encoder"
    $firstFrameSource = if ([string]::IsNullOrWhiteSpace($FirstFrame)) { $null } else { Resolve-ExistingFile $FirstFrame "First-frame image" }
    $lastFrameSource = if ([string]::IsNullOrWhiteSpace($LastFrame)) { $null } else { Resolve-ExistingFile $LastFrame "Last-frame image" }
    $firstFrame = $firstFrameSource
    $lastFrame = $lastFrameSource
    if (($null -eq $firstFrame) -and ($null -eq $lastFrame)) {
        $mode = "t2v"
    } else {
        $mode = "fl2va-i2v"
        if ($Width -lt 64 -or $Height -lt 64) {
            throw "I2V width and height must both be at least 64"
        }
    }
    $repositoryRoot = ConvertTo-AbsolutePath (Join-Path $PSScriptRoot "..")
    $helper = Resolve-ExistingFile (Join-Path $repositoryRoot "scripts\encode_h3_quantized_prompt.py") "conditioning helper"
    $python = Resolve-ComfyPython $comfyRoot $ComfyPython
    $binary = Resolve-H3Binary $repositoryRoot $BinaryPath
    $output = ConvertTo-AbsolutePath $Output
    if ([string]::IsNullOrWhiteSpace($SidecarPath)) {
        $sidecar = "$output.conditioning.h3c"
    } else {
        $sidecar = ConvertTo-AbsolutePath $SidecarPath
    }
    if ([StringComparer]::OrdinalIgnoreCase.Equals($output, $sidecar)) {
        throw "Output and conditioning sidecar must be different paths"
    }
    if (-not $sidecar.EndsWith(".h3c", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Conditioning sidecar path must end in .h3c"
    }

    $helperArguments = @(
        $helper,
        "--comfyui", $comfyRoot,
        "--text-encoder", $encoder,
        "--output", $sidecar,
        "--prompt", $Prompt,
        "--device", $Device,
        "--mode", $mode,
        "--width", $Width.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--height", $Height.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($null -ne $firstFrame) { $helperArguments += @("--first-frame", $firstFrame) }
    if ($null -ne $lastFrame) { $helperArguments += @("--last-frame", $lastFrame) }
    Write-Host "[h3cspeed] generating GPU ComfyUI conditioning sidecar"
    $helperLines = @(& $python @helperArguments 2>&1 | ForEach-Object { $_.ToString() })
    $helperExitCode = $LASTEXITCODE
    foreach ($line in $helperLines) {
        Write-Host "[conditioning] $line"
    }
    if ($helperExitCode -ne 0) {
        throw "conditioning helper failed with exit code $helperExitCode"
    }
    if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
        throw "conditioning helper succeeded but did not create sidecar: $sidecar"
    }
    if ($null -ne $firstFrameSource) {
        $firstFrame = Resolve-ExistingFile "$sidecar.first.png" "Canonical first-frame image"
    }
    if ($null -ne $lastFrameSource) {
        $lastFrame = Resolve-ExistingFile "$sidecar.last.png" "Canonical last-frame image"
    }

    $reportedHashes = @(
        [regex]::Matches(($helperLines -join "`n"), "model_sha256=([0-9a-fA-F]{64})") |
            ForEach-Object { $_.Groups[1].Value.ToLowerInvariant() }
    )
    $reportedHashes = @($reportedHashes | Select-Object -Unique)
    if ($reportedHashes.Count -gt 1) {
        throw "conditioning helper reported multiple model SHA-256 values"
    }
    $computedHash = (Get-FileHash -LiteralPath $encoder -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($reportedHashes.Count -eq 1 -and $reportedHashes[0] -ne $computedHash) {
        throw "conditioning helper SHA-256 does not match the text encoder file"
    }
    $encoderHash = if ($reportedHashes.Count -eq 1) { $reportedHashes[0] } else { $computedHash }

    # The native runtime validates prompt, token IDs, recipe and this whole-file
    # fingerprint before committing the sidecar tensor.  Keep this bridge
    # process-local and restore any caller values in finally.
    $env:H3CSPEED_TEXT_EMBEDDING = $sidecar
    $env:H3CSPEED_TEXT_ENCODER_SHA256 = $encoderHash
    $cliArguments = @(
        "-d", $modelRoot,
        "-p", $Prompt,
        "--width", $Width.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--height", $Height.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--frames", $Frames.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--steps", $Steps.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--layers", $Layers.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--reuse", $Reuse.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--core-reuse", $CoreReuse.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--ssd-streaming",
        "-o", $output
    )
    if ($null -ne $firstFrame) { $cliArguments += @("--first-frame", $firstFrame) }
    if ($null -ne $lastFrame) { $cliArguments += @("--last-frame", $lastFrame) }
    Write-Host "[h3cspeed] running hybrid Comfy-conditioned/native INT8 DiT path"
    & $binary @cliArguments
    $exitCode = $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("run-h3-quantized: $($_.Exception.Message)")
} finally {
    Restore-ProcessEnvironmentVariable "H3CSPEED_TEXT_EMBEDDING" $oldSidecar
    Restore-ProcessEnvironmentVariable "H3CSPEED_TEXT_ENCODER_SHA256" $oldEncoderSha
}

if ($exitCode -ne 0) {
    exit $exitCode
}
exit 0
