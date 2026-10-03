<#
.SYNOPSIS
    Starts the creditsurv API and its dashboard in WSL (Linux), in one command.

.DESCRIPTION
    Windows Smart App Control blocks lightgbm and shap on this machine, so the app
    has to be served from Linux. This script:

      1. stops any Windows process listening on the port (default 8501), so a stale
         Windows Streamlit cannot answer in place of the Linux one;
      2. syncs the code from this folder to ~/creditsurv in WSL (src, app, scripts,
         config, tests, .streamlit, FINDINGS.md, README.md; never .venv), and
         outputs/models and outputs/data only where the Windows file is newer;
      3. checks in WSL that lifelines, lightgbm, shap, streamlit and psutil import,
         and names anything missing with the command that fixes it;
      4. starts the API (uvicorn, port 8000) and, as its client, the Streamlit
         dashboard (port 8501) in WSL, and prints both URLs. Ctrl+C stops both.

    The Linux half is scripts/wsl_launch.sh. No Windows security setting is read or
    changed.

.PARAMETER Port
    Port to serve on. Default 8501.

.PARAMETER ApiPort
    Port for the API. Default 8000.

.PARAMETER ApiOnly
    Start the API only (no dashboard): for scripts or another client. Holds this
    window until Ctrl+C.

.PARAMETER ApiBackground
    Start the API only and return: it keeps serving after this window closes.
    Stop it with the pid the script prints (also in outputs/logs/api.pid).

.PARAMETER CheckOnly
    Stop after the sync and the import check; do not start the app.

.PARAMETER CopyBack
    Instead of starting the app: copy finished results from ~/creditsurv/outputs
    back to this folder's outputs (newer files only), so they show in File Explorer
    and VS Code. Unfinished runs are skipped and named.

.PARAMETER Distro
    WSL distribution to use. Default: the WSL default distribution.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_linux.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_linux.ps1 -CopyBack
#>
[CmdletBinding()]
param(
    [int]$Port = 8501,
    [int]$ApiPort = 8000,
    [switch]$ApiOnly,
    [switch]$ApiBackground,
    [switch]$CheckOnly,
    [switch]$CopyBack,
    [string]$Distro = ""
)

$Root = $PSScriptRoot
$WslArgs = @()
if ($Distro) { $WslArgs += @("-d", $Distro) }

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Host "FAILED: WSL is not installed. Install it with: wsl --install -d Ubuntu" -ForegroundColor Red
    exit 1
}

# Paths are converted by WSL itself, so a folder with spaces (or on OneDrive) works.
$RootWsl = & wsl.exe @WslArgs -e wslpath -a "$Root"
if ($LASTEXITCODE -ne 0 -or -not $RootWsl) {
    Write-Host "FAILED: WSL did not start or could not see $Root." -ForegroundColor Red
    Write-Host "  Try 'wsl -l -v' to check the distribution, then run this again." -ForegroundColor Red
    exit 1
}
$RootWsl = "$RootWsl".Trim()
$Script = "$RootWsl/scripts/wsl_launch.sh"

if ($CopyBack) {
    & wsl.exe @WslArgs -e bash $Script copy-back $RootWsl $Port
    exit $LASTEXITCODE
}

# ------------------------------------------- free the port on Windows -------
# WSL's own port forwarder (wslrelay) shows up as the listener when an earlier
# Linux Streamlit is still running; that one is stopped from the Linux side instead,
# because stopping the forwarder would break localhost access to WSL. System
# processes are never touched.
$Infrastructure = @("wslrelay", "wslhost", "wslservice", "vmmem", "vmmemWSL", "svchost", "System", "Idle")
$Listeners = Get-NetTCPConnection -LocalPort @($Port, $ApiPort) -State Listen -ErrorAction SilentlyContinue |
             Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $Listeners) {
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    if ($Infrastructure -contains $proc.ProcessName) {
        Write-Host "Port ${Port}: held by $($proc.ProcessName) (WSL/system); handled on the Linux side." -ForegroundColor DarkGray
        continue
    }
    try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        Write-Host "Stopped Windows process on port ${Port}: $($proc.ProcessName) (pid $procId) $($proc.Path)" -ForegroundColor Yellow
    } catch {
        Write-Host "FAILED: could not stop $($proc.ProcessName) (pid $procId) on port ${Port}: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  Close it yourself, or start on another port: .\run_linux.ps1 -Port 8502" -ForegroundColor Red
        exit 1
    }
}
# Give Windows a moment to release the socket.
for ($i = 0; $i -lt 10; $i++) {
    $still = Get-NetTCPConnection -LocalPort @($Port, $ApiPort) -State Listen -ErrorAction SilentlyContinue |
             Where-Object { $Infrastructure -notcontains (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName }
    if (-not $still) { break }
    Start-Sleep -Milliseconds 300
}

# ------------------------------------------------ sync, check, serve --------
$Mode = if ($CheckOnly) { "check" }
        elseif ($ApiBackground) { "api-bg" }
        elseif ($ApiOnly) { "api" }
        else { "serve" }
& wsl.exe @WslArgs -e env "CREDITSURV_API_PORT=$ApiPort" bash $Script $Mode $RootWsl $Port
exit $LASTEXITCODE
