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
    [int]$RenderWidth = 0,
    [int]$RenderHeight = 0,
    [int]$Frames = 22,
    [int]$Layers = 50,
    [int]$Reuse = 1,
    [int]$CoreReuse = 1,
    [UInt64]$Seed = 42,
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
$oldCudaDevice = [Environment]::GetEnvironmentVariable("H3_CUDA_DEVICE", "Process")

try {
    if ([string]::IsNullOrWhiteSpace($Prompt)) {
        throw "Prompt must not be empty"
    }
    if ($Steps -lt 2 -or $Steps -gt 100) {
        throw "Steps must be between 2 and 100"
    }
    if ($Width -lt 64 -or $Height -lt 64) {
        throw "Width and Height must be at least 64"
    }
    if (($Width % 32) -ne 0 -or ($Height % 32) -ne 0) {
        throw "Width and Height must be divisible by 32"
    }
    if ([int64]$Width * [int64]$Height -gt 768 * 1344) {
        throw "Output canvas exceeds the released 768x1344 pixel limit"
    }
    if (($RenderWidth -eq 0) -xor ($RenderHeight -eq 0)) {
        throw "RenderWidth and RenderHeight must be supplied together"
    }
    if ($RenderWidth -ne 0) {
        if ($RenderWidth -lt 64 -or $RenderHeight -lt 64) {
            throw "RenderWidth and RenderHeight must be at least 64"
        }
        if ($RenderWidth -gt $Width -or $RenderHeight -gt $Height) {
            throw "Render dimensions must not exceed output dimensions"
        }
        if (($RenderWidth % 32) -ne 0 -or ($RenderHeight % 32) -ne 0) {
            throw "RenderWidth and RenderHeight must be divisible by 32"
        }
        if (($RenderWidth * $Height) -ne ($RenderHeight * $Width)) {
            throw "Render dimensions must preserve the output aspect ratio"
        }
    }
    $effectiveRenderWidth = if ($RenderWidth -eq 0) { $Width } else { $RenderWidth }
    $effectiveRenderHeight = if ($RenderHeight -eq 0) { $Height } else { $RenderHeight }
    if ($Frames -lt 22 -or $Frames -gt 362 -or (($Frames - 5) % 17) -ne 0) {
        throw "Frames must follow the released 5 + 17n layout within 22..362"
    }
    if ($Layers -lt 1 -or $Layers -gt 50) {
        throw "Layers must be between 1 and 50"
    }
    if ($Reuse -lt 1 -or $Reuse -gt 3 -or $CoreReuse -lt 1 -or $CoreReuse -gt 6) {
        throw "Reuse must be in 1..3 and CoreReuse must be in 1..6"
    }
    if ($Reuse -gt 1 -and $CoreReuse -gt 1) {
        throw "Reuse > 1 and CoreReuse > 1 cannot be combined"
    }
    if ([string]::IsNullOrWhiteSpace($Device) -or -not [regex]::IsMatch($Device, '^cuda(?::[0-9]+)?$', [Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
        throw "Device must be a CUDA device (for example cuda:0); CPU fallback is forbidden"
    }

    $modelRoot = Resolve-ExistingDirectory $ModelRoot "Model root"
    $manifest = $null
    $manifestPath = Join-Path (Split-Path -Parent $modelRoot) "manifest.json"
    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if ($null -ne $manifest.model_family -and $manifest.model_family -ne "FL2VA") {
            throw "Quantized model manifest is not an FL2VA pack: $manifestPath"
        }
        if ($null -ne $manifest.unsupported_model_families -and
            @($manifest.unsupported_model_families) -contains "Ref2VA" -and
            (Test-Path -LiteralPath (Join-Path $modelRoot "Ref2VA") -PathType Container)) {
            throw "Prepared FL2VA pack unexpectedly contains Ref2VA payloads"
        }
    }
    $comfyRoot = Resolve-ExistingDirectory $ComfyUIRoot "ComfyUI root"
    $encoder = Resolve-ExistingFile $TextEncoder "Quantized Qwen text encoder"
    $firstFrameSource = if ([string]::IsNullOrWhiteSpace($FirstFrame)) { $null } else { Resolve-ExistingFile $FirstFrame "First-frame image" }
    $lastFrameSource = if ([string]::IsNullOrWhiteSpace($LastFrame)) { $null } else { Resolve-ExistingFile $LastFrame "Last-frame image" }
    $canonicalFirstFrame = $firstFrameSource
    $canonicalLastFrame = $lastFrameSource
    if (($null -eq $canonicalFirstFrame) -and ($null -eq $canonicalLastFrame)) {
        $mode = "t2v"
    } else {
        $mode = "fl2va-i2v"
        if ($null -ne $manifest -and $null -ne $manifest.capabilities) {
            $requiredCapability = if (($null -ne $canonicalFirstFrame) -and ($null -ne $canonicalLastFrame)) {
                "fl2va_i2v_first_and_last_frames"
            } elseif ($null -ne $canonicalFirstFrame) {
                "fl2va_i2v_first_frame"
            } else {
                "fl2va_i2v_last_frame"
            }
            if (@($manifest.capabilities) -notcontains $requiredCapability) {
                throw "Quantized FL2VA manifest does not declare $requiredCapability"
            }
        }
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
        "--width", $effectiveRenderWidth.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--height", $effectiveRenderHeight.ToString([Globalization.CultureInfo]::InvariantCulture)
    )
    if ($null -ne $canonicalFirstFrame) { $helperArguments += @("--first-frame", $canonicalFirstFrame) }
    if ($null -ne $canonicalLastFrame) { $helperArguments += @("--last-frame", $canonicalLastFrame) }
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
        $canonicalFirstFrame = Resolve-ExistingFile "$sidecar.first.png" "Canonical first-frame image"
    }
    if ($null -ne $lastFrameSource) {
        $canonicalLastFrame = Resolve-ExistingFile "$sidecar.last.png" "Canonical last-frame image"
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
    $env:H3_CUDA_DEVICE = if ($Device.Contains(":")) {
        $Device.Substring($Device.IndexOf(":") + 1)
    } else {
        "0"
    }
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
        "--seed", $Seed.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--ssd-streaming",
        "-o", $output
    )
    if ($RenderWidth -ne 0) {
        $cliArguments += @(
            "--render-width", $RenderWidth.ToString([Globalization.CultureInfo]::InvariantCulture),
            "--render-height", $RenderHeight.ToString([Globalization.CultureInfo]::InvariantCulture)
        )
    }
    if ($null -ne $firstFrameSource) {
        if ([string]::IsNullOrWhiteSpace($canonicalFirstFrame)) { throw "Canonical first-frame path is empty" }
        $cliArguments += @("--first-frame", $canonicalFirstFrame)
        Write-Host "[h3cspeed] native canonical first-frame: $canonicalFirstFrame"
    }
    if ($null -ne $lastFrameSource) {
        if ([string]::IsNullOrWhiteSpace($canonicalLastFrame)) { throw "Canonical last-frame path is empty" }
        $cliArguments += @("--last-frame", $canonicalLastFrame)
        Write-Host "[h3cspeed] native canonical last-frame: $canonicalLastFrame"
    }
    Write-Host "[h3cspeed] running hybrid Comfy-conditioned/native INT8 DiT path"
    & $binary @cliArguments
    $exitCode = $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("run-h3-quantized: $($_.Exception.Message)")
} finally {
    Restore-ProcessEnvironmentVariable "H3CSPEED_TEXT_EMBEDDING" $oldSidecar
    Restore-ProcessEnvironmentVariable "H3CSPEED_TEXT_ENCODER_SHA256" $oldEncoderSha
    Restore-ProcessEnvironmentVariable "H3_CUDA_DEVICE" $oldCudaDevice
}

if ($exitCode -ne 0) {
    exit $exitCode
}
exit 0
