<#
.SYNOPSIS
    Runs the out-of-time holdout sequence (train 2007-2015, test 2016-2018) end to end.

.DESCRIPTION
    Pure orchestration: calls the existing stage scripts in order and stops at the
    first failure. It does not change what any script does.

      -Size Small   dev sample (200k loans), minimal SHAP work.        ~3-6 min
      -Size Medium  dev sample, production SHAP settings, moderate n.  ~12-20 min
      -Size Full    full data, the pre-registered design exactly.      ~35-55 min

    Small and Medium write under their own tags and only PREVIEW FINDINGS section 6
    (05_report --dry-run), so a test run can never fill the pre-registered section or
    block the real run. Only -Size Full writes FINDINGS.md.

.PARAMETER Size
    Small (default), Medium or Full.

.PARAMETER PlanOnly
    Print the stage commands that would run, then exit without running anything.

.PARAMETER Overwrite
    Pass --overwrite to every stage. Off by default: re-running a size whose
    outputs already exist stops at the first stage with a clear message instead.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_holdout.ps1 -Size Small
#>
[CmdletBinding()]
param(
    [ValidateSet("Small", "Medium", "Full")]
    [string]$Size = "Small",
    [switch]$Overwrite,
    [switch]$PlanOnly
)

$Root = $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$VenvDir = Join-Path $Root ".venv"

# ---------------------------------------------------------------- venv --------
# Activate only if no venv is active. If a DIFFERENT venv is active, say so; the
# stages are always run with this project's interpreter ($Python) regardless, so a
# stray activated environment can never run them.
if (-not $env:VIRTUAL_ENV) {
    . (Join-Path $VenvDir "Scripts\Activate.ps1")
    Write-Host "venv activated: $VenvDir"
} elseif ((Resolve-Path $env:VIRTUAL_ENV).Path -ne (Resolve-Path $VenvDir).Path) {
    Write-Host "NOTE: another venv is active ($env:VIRTUAL_ENV); stages will use $Python" -ForegroundColor Yellow
} else {
    Write-Host "venv already active: $env:VIRTUAL_ENV"
}
if (-not (Test-Path $Python)) {
    Write-Host "FAILED: project interpreter not found at $Python" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------- stage plan ---------
# The command list comes from creditsurv.plan, the single definition shared with
# the Streamlit UI, so the two cannot drift. tests/test_plan.py pins it to exactly
# what this wrapper ran before the switch (tests/fixtures/wrapper_commands_pinned.json).
$SrcDir = Join-Path $Root "src"
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$SrcDir;$env:PYTHONPATH" } else { $env:PYTHONPATH = $SrcDir }
$PlanArgs = @("-m", "creditsurv.plan", "--size", $Size, "--json")
if ($Overwrite) { $PlanArgs += "--overwrite" }
$PlanJson = & $Python @PlanArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: could not build the stage plan (creditsurv.plan exit $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}
$Plan = ($PlanJson -join "`n") | ConvertFrom-Json
$Tag = $Plan.tag
$StratTag = $Plan.strat_tag
$Stages = @($Plan.stages)

if ($PlanOnly) {
    # Print the commands that WOULD run, then stop. Nothing is executed.
    foreach ($s in $Stages) {
        Write-Output ("STAGE|" + $s.name + "|" + $s.tag + "|" + ($s.args -join [char]0x1f))
    }
    exit 0
}

# ----------------------------------------------------------------- run --------
$Total = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host ""
Write-Host ("=" * 78)
Write-Host "HOLDOUT SEQUENCE  size=$Size  tag=$Tag  strat-tag=$StratTag  overwrite=$([bool]$Overwrite)"
Write-Host ("=" * 78)

$i = 0
foreach ($s in $Stages) {
    $i++
    $label = "[$i/$($Stages.Count)] $($s.Name)"
    Write-Host ""
    Write-Host ("-" * 78)
    Write-Host "$label   (--tag $($s.Tag))   started $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "  > python $($s.Args -join ' ')" -ForegroundColor DarkGray
    Write-Host ("-" * 78)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python @($s.Args)
    $code = $LASTEXITCODE
    $mins = "{0:N1}" -f $sw.Elapsed.TotalMinutes
    if ($code -ne 0) {
        Write-Host ""
        Write-Host "$label  FAILED (exit code $code) after $mins min, see output above." -ForegroundColor Red
        if ($code -eq 4) {
            Write-Host "  Exit 4 = the stage refused to overwrite existing outputs for this tag." -ForegroundColor Red
        }
        Write-Host "  Stopped. No later stage was run and nothing was cleaned up or retried." -ForegroundColor Red
        exit $code
    }
    Write-Host "$label  completed in $mins min" -ForegroundColor Green
}

# ------------------------------------------------- FINDINGS.md diff -----------
Write-Host ""
Write-Host ("=" * 78)
Write-Host "git diff FINDINGS.md (against the last commit)"
Write-Host ("=" * 78)
$diff = git --no-pager diff -- FINDINGS.md
if (-not $diff) {
    Write-Host "No changes to FINDINGS.md." -ForegroundColor Green
} else {
    git --no-pager diff --stat -- FINDINGS.md
    $diff | ForEach-Object { Write-Host $_ }
    # Flag any change that starts before the section 6 heading of the committed file:
    # sections 0-5 hold the primary results and must never change in a holdout run.
    $h6 = (git show HEAD:FINDINGS.md | Select-String -Pattern '^## 6\. ' | Select-Object -First 1).LineNumber
    $early = git --no-pager diff -U0 -- FINDINGS.md |
             Select-String -Pattern '^@@ -(\d+)' |
             Where-Object { $h6 -and [int]$_.Matches[0].Groups[1].Value -lt $h6 }
    if ($early) {
        Write-Host ""
        Write-Host "WARNING: FINDINGS.md changed ABOVE section 6 (committed heading at line $h6)." -ForegroundColor Red
        Write-Host "         The primary-result sections were touched. Review before committing." -ForegroundColor Red
    } else {
        Write-Host ""
        Write-Host "All changes are in section 6 (at or below committed line $h6); sections 0-5 untouched." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Sequence complete: size=$Size, total $("{0:N1}" -f $Total.Elapsed.TotalMinutes) min." -ForegroundColor Green
