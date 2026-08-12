[CmdletBinding()]
param(
    [string]$RepositoryRoot = '',
    [string]$ModelRoot = 'E:\models',
    [ValidateRange(1, 64)]
    [int]$Workers = 2,
    [Alias('Python')]
    [string]$PythonPath = 'python'
)

$ErrorActionPreference = 'Stop'

# Resolve paths from the script location, not the caller's current directory.
$scriptPath = Join-Path $PSScriptRoot 'download_h3_fl2va.py'
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
if ($env:H3_MODEL_ROOT -and $ModelRoot -eq 'E:\models') {
    $ModelRoot = $env:H3_MODEL_ROOT
}
if ($env:H3_PYTHON -and $PythonPath -eq 'python') {
    $PythonPath = $env:H3_PYTHON
}

# One model root is the path contract shared by the launcher and Python.
$cacheRoot = Join-Path $ModelRoot 'hf-cache'
$homeRoot = Join-Path $ModelRoot 'hf-home'
$xetRoot = Join-Path $ModelRoot 'hf-xet'
$tempRoot = Join-Path $ModelRoot 'hf-tmp'
$modelDir = Join-Path $ModelRoot 'MiniMax-H3'
$runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$runId = [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runStem = Join-Path $ModelRoot ("download-h3-fl2va-{0}-{1}" -f $runStamp,$runId)
$logPath = "$runStem.log"
$errorPath = "$runStem.err"
$statusPath = "$runStem.status.json"

New-Item -ItemType Directory -Force -Path `
    $ModelRoot,$cacheRoot,$homeRoot,$xetRoot,$tempRoot,$modelDir | Out-Null

# Pass every derived path explicitly as a coherent set. Python also has the
# same E: defaults for direct invocation, while these values win when this
# launcher is used with a custom model volume.
$env:H3_MODEL_ROOT = $ModelRoot
$env:H3_LOCAL_DIR = $modelDir
$env:H3_CACHE_DIR = $cacheRoot
$env:H3_TEMP_DIR = $tempRoot
$env:H3_XET_CACHE = $xetRoot
$env:H3_HF_HOME = $homeRoot
$env:HF_HOME = $homeRoot
$env:HF_HUB_CACHE = $cacheRoot
$env:HF_XET_CACHE = $xetRoot
$env:HF_DATASETS_CACHE = Join-Path $cacheRoot 'datasets'
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:H3_DOWNLOAD_WORKERS = [string]$Workers
$env:H3_STATUS_PATH = $statusPath
$env:H3_LOG_PATH = $logPath
$env:H3_ERROR_PATH = $errorPath
$env:H3_LOCK_PATH = Join-Path $ModelRoot '.download-h3-fl2va.lock'

$process = Start-Process -FilePath $PythonPath `
    -ArgumentList @('-u', $scriptPath) `
    -WorkingDirectory $RepositoryRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errorPath `
    -PassThru

# This command only starts a background job. Completion belongs in the
# timestamped status JSON and must not be inferred from Start-Process.
Write-Output ("started pid={0} model={1} workers={2} log={3} err={4} status={5}" -f `
    $process.Id,$modelDir,$Workers,$logPath,$errorPath,$statusPath)
