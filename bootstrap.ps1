<#
.SYNOPSIS
    One-command setup for qwenstyle (Windows, PowerShell 5.1+).

.DESCRIPTION
    Creates .venv, installs this package with CUDA 12.9 torch wheels, prints the model plan of a
    machine profile, downloads those repositories from HuggingFace and runs a self-test.
    The script is idempotent: an existing .venv and already installed packages are reused, and
    nothing is deleted.

    Profile repository lists live in qwenstyle/profiles.py only - this script never duplicates them.

.EXAMPLE
    cmd /c bootstrap.cmd -Profile strong -DryRun
    cmd /c bootstrap.cmd -Profile strong -Check
    cmd /c bootstrap.cmd -Profile strong -SkipModels
#>
[CmdletBinding()]
param(
    [ValidateSet('weak', 'strong', 'flagship')]
    [string]$Profile = 'strong',
    [switch]$DryRun,
    [switch]$Check,
    [switch]$SkipModels,
    [switch]$NoVenv
)

$ErrorActionPreference = 'Stop'
$Total = 9
$script:Step = 0
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

function Write-Step([string]$text) {
    $script:Step = $script:Step + 1
    Write-Host ''
    Write-Host ('[' + $script:Step + '/' + $Total + '] ' + $text) -ForegroundColor Cyan
}

function Write-Note([string]$text) {
    Write-Host ('    ' + $text) -ForegroundColor DarkGray
}

function Write-Warn([string]$text) {
    Write-Host ('    ! ' + $text) -ForegroundColor Yellow
}

function Fail([string]$message, [int]$code) {
    Write-Host ''
    Write-Host ('[error] ' + $message) -ForegroundColor Red
    exit $code
}

# Runs a native command, echoes it, sends its output straight to the console (never into the
# pipeline: a function's return value must stay the exit code) and returns that exit code.
function Invoke-Native([string]$exe, [string[]]$cmdArgs) {
    Write-Note ('> ' + $exe + ' ' + ($cmdArgs -join ' '))
    & $exe @cmdArgs | Out-Host
    if ($null -eq $LASTEXITCODE) { return 0 }
    return [int]$LASTEXITCODE
}

function Test-Python312($candidate) {
    $probe = @($candidate.Args) + @('-c', 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)')
    try { & $candidate.Exe @probe *> $null } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}

# `py -3.12`, then `python`, then `python3` - the first one that reports Python 3.12 wins.
function Find-Python312 {
    foreach ($name in @('py', 'python', 'python3')) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -eq $found) { continue }
        $exePath = $found.Source
        if (-not $exePath) { $exePath = $found.Name }
        $prefix = @()
        if ($name -eq 'py') { $prefix = @('-3.12') }
        $candidate = New-Object psobject -Property @{ Exe = $exePath; Args = $prefix }
        if (Test-Python312 $candidate) { return $candidate }
    }
    return $null
}

# torch/diffusers/transformers/qwenstyle importable by this interpreter -> install can be skipped.
function Test-EnvReady($py) {
    $code = "import importlib.util as u, sys; names = ('torch', 'diffusers', 'transformers', 'qwenstyle'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)"
    try { & $py.Exe (@($py.Args) + @('-c', $code)) *> $null } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}

# Snippet prefix that loads the manifest by path: profiles.py imports stdlib only, so this works
# even before the package dependencies are installed. Single quotes only - PowerShell 5.1 native
# argument passing eats embedded double quotes.
$script:ManifestExpr = "import importlib.util as u, sys; s = u.spec_from_file_location('_qp', 'qwenstyle/profiles.py'); m = u.module_from_spec(s); sys.modules['_qp'] = m; s.loader.exec_module(m); p = m.profile(sys.argv[1]); "

# First stdout line of a snippet, or '' when the interpreter/import fails.
function Invoke-Snippet($py, [string]$code, [string]$name) {
    if ($null -eq $py) { return '' }
    $lines = @()
    try { $lines = @(& $py.Exe (@($py.Args) + @('-c', $code, $name)) 2>$null) } catch { return '' }
    if ($LASTEXITCODE -ne 0) { return '' }
    return (($lines | ForEach-Object { [string]$_ }) -join ' ').Trim()
}

# Hardware line from qwenstyle/profiles.py (never duplicated here).
function Get-ProfileLine($py, [string]$name) {
    $code = $script:ManifestExpr + "print(p.hardware + ', ~%.1f GiB on disk' % p.disk_gib)"
    return Invoke-Snippet $py $code $name
}

# Presets addressed to this profile (also from the manifest).
function Get-ProfilePresets($py, [string]$name) {
    $code = $script:ManifestExpr + "print(', '.join(p.presets))"
    return Invoke-Snippet $py $code $name
}

# Fallback plan when the CLI is not importable yet: same manifest, printed with plain Python.
function Show-ProfilePlanFallback($py, [string]$name) {
    if ($null -eq $py) { return }
    $code = $script:ManifestExpr +
            "print('  download %.1f GiB, on disk %.1f GiB' % (m.download_estimate(p.name), p.disk_gib)); " +
            "[print('  %-52s %-34s %-6s %s' % (r.repo, r.license or '-', 'gated' if r.gated else '-', r.note)) for r in p.repos]"
    & $py.Exe (@($py.Args) + @('-c', $code, $name)) | Out-Host
}

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #

Write-Host ''
Write-Host 'qwenstyle bootstrap - local anime style-transfer and generation' -ForegroundColor White
Write-Host '18+ / NSFW: adult anime content; model licences belong to their HuggingFace repositories.' -ForegroundColor Gray
Write-Note ('root:    ' + $Root)
Write-Note ('profile: ' + $Profile)
if ($DryRun) {
    Write-Note 'mode:    -DryRun (prints the plan, installs nothing, downloads nothing)'
} elseif ($Check) {
    Write-Note 'mode:    -Check (setup + Hub access probe, no full download)'
} else {
    Write-Note 'mode:    full setup'
}

# --------------------------------------------------------------------------- #
# 1/9 interpreter and virtual environment
# --------------------------------------------------------------------------- #

Write-Step 'Python 3.12 and virtual environment'

$basePython = Find-Python312
$venvPython = Join-Path $Root '.venv\Scripts\python.exe'
$python = $null

if ($NoVenv) {
    if ($null -eq $basePython) {
        if (-not $DryRun) { Fail 'python 3.12 not found: install Python 3.12 from python.org or drop -NoVenv' 2 }
        Write-Warn 'python 3.12 not found (preview only: setup steps are printed, not run)'
    } else {
        $python = $basePython
        Write-Note ('-NoVenv: using ' + $python.Exe + ' ' + ($python.Args -join ' '))
    }
} elseif (Test-Path -LiteralPath $venvPython) {
    $python = New-Object psobject -Property @{ Exe = $venvPython; Args = @() }
    Write-Note ('using existing .venv: ' + $venvPython)
} elseif ($null -eq $basePython) {
    if (-not $DryRun) { Fail 'python 3.12 not found: install Python 3.12 from python.org, then re-run bootstrap.cmd' 2 }
    Write-Warn 'python 3.12 not found (preview only: setup steps are printed, not run)'
} elseif ($DryRun) {
    Write-Note ('would create .venv with: ' + $basePython.Exe + ' ' + (($basePython.Args + @('-m', 'venv', '.venv')) -join ' '))
    $python = $basePython
} else {
    Write-Note ('creating .venv with: ' + $basePython.Exe + ' ' + (($basePython.Args + @('-m', 'venv', '.venv')) -join ' '))
    $code = Invoke-Native $basePython.Exe (@($basePython.Args) + @('-m', 'venv', '.venv'))
    if ($code -ne 0) { Fail ('python -m venv .venv failed with exit code ' + $code) $code }
    if (-not (Test-Path -LiteralPath $venvPython)) { Fail '.venv\Scripts\python.exe was not created' 2 }
    $python = New-Object psobject -Property @{ Exe = $venvPython; Args = @() }
}

$profileLine = Get-ProfileLine $python $Profile
if ($profileLine) { Write-Note ('profile hardware: ' + $profileLine) }

# --------------------------------------------------------------------------- #
# 2/9 packages (torch with CUDA wheels first, then this package)
# --------------------------------------------------------------------------- #

Write-Step 'Python packages: torch (CUDA 12.9 wheels), then this package'

$pipUpdateArgs = @('-m', 'pip', 'install', '--upgrade', 'pip')
$torchArgs = @('-m', 'pip', 'install', 'torch', 'torchvision',
               '--index-url', 'https://download.pytorch.org/whl/cu129')
$editableArgs = @('-m', 'pip', 'install', '-e', '.')

if ($null -eq $python) {
    Write-Note 'skipped: no Python 3.12 interpreter'
} elseif ($DryRun) {
    Write-Note 'would run (torch first: plain pip on Windows would pull a CPU-only build):'
    Write-Note ('  ' + $python.Exe + ' ' + (($python.Args + $pipUpdateArgs) -join ' '))
    Write-Note ('  ' + $python.Exe + ' ' + (($python.Args + $torchArgs) -join ' '))
    Write-Note ('  ' + $python.Exe + ' ' + (($python.Args + $editableArgs) -join ' '))
} elseif (Test-EnvReady $python) {
    Write-Note 'already installed (torch/diffusers/transformers/qwenstyle importable), skipping pip'
} else {
    $code = Invoke-Native $python.Exe (@($python.Args) + $pipUpdateArgs)
    if ($code -ne 0) { Fail ('pip upgrade failed with exit code ' + $code) $code }
    $code = Invoke-Native $python.Exe (@($python.Args) + $torchArgs)
    if ($code -ne 0) { Fail ('torch install failed with exit code ' + $code + ' (CUDA 12.9 wheels need Python 3.12 on win64)') $code }
    $code = Invoke-Native $python.Exe (@($python.Args) + $editableArgs)
    if ($code -ne 0) { Fail ('pip install -e . failed with exit code ' + $code) $code }
}

# --------------------------------------------------------------------------- #
# 3/9 model plan (always printed, never downloads)
# --------------------------------------------------------------------------- #

Write-Step ('Model plan for profile ' + $Profile + ' (no download; gated repositories are marked)')
if ($null -eq $python) {
    Write-Note 'skipped: no Python 3.12 interpreter'
} else {
    $code = Invoke-Native $python.Exe (@($python.Args) + @('-m', 'qwenstyle', 'models', 'download',
                                                          '--profile', $Profile, '--dry-run'))
    if ($code -ne 0) {
        Write-Warn ('plan command failed with exit code ' + $code + '; showing the manifest directly')
        Show-ProfilePlanFallback $python $Profile
    }
}

# --------------------------------------------------------------------------- #
# 4/9 -DryRun stops here
# --------------------------------------------------------------------------- #

if ($DryRun) {
    Write-Step 'Preview finished (-DryRun): nothing installed, nothing downloaded'
    Write-Note 'run without -DryRun to install; add -Check to stop after the Hub access probe'
    Write-Host ''
    exit 0
}

# --------------------------------------------------------------------------- #
# 5/9 -Check: Hub access probe, one small file per repository
# --------------------------------------------------------------------------- #

if ($Check) {
    Write-Step ('Hub access probe for profile ' + $Profile + ' (one small file per repository, no weights)')
    $previousOffline = $env:HF_HUB_OFFLINE
    $env:HF_HUB_OFFLINE = '0'
    try {
        $code = Invoke-Native $python.Exe (@($python.Args) + @('-m', 'qwenstyle', 'models', 'download',
                                                              '--profile', $Profile, '--probe',
                                                              '--probe-mib', '1', '--cleanup'))
    } finally {
        $env:HF_HUB_OFFLINE = $previousOffline
    }
    if ($code -eq 0) {
        Write-Host ''
        Write-Host '    check passed: every repository of this profile is reachable' -ForegroundColor Green
    } else {
        Write-Warn ('check failed with exit code ' + $code + ' - see the FAIL rows above (gated repositories need HF_TOKEN)')
    }
    exit $code
}

# --------------------------------------------------------------------------- #
# 6/9 download weights
# --------------------------------------------------------------------------- #

Write-Step ('Download model weights for profile ' + $Profile)
if ($SkipModels) {
    Write-Note 'skipped: -SkipModels'
} else {
    Write-Note 'only the repositories of this profile are fetched, under their own licences'
    $previousOffline = $env:HF_HUB_OFFLINE
    $env:HF_HUB_OFFLINE = '0'
    try {
        $code = Invoke-Native $python.Exe (@($python.Args) + @('-m', 'qwenstyle', 'models', 'download',
                                                              '--profile', $Profile, '--yes'))
    } finally {
        $env:HF_HUB_OFFLINE = $previousOffline
    }
    if ($code -ne 0) { Fail ('models download failed with exit code ' + $code) $code }
}

# --------------------------------------------------------------------------- #
# 7/9 environment report
# --------------------------------------------------------------------------- #

Write-Step 'Environment report'
$code = Invoke-Native $python.Exe (@($python.Args) + @('-m', 'qwenstyle', 'env'))
if ($code -ne 0) { Fail ('qwenstyle env failed with exit code ' + $code) $code }

# --------------------------------------------------------------------------- #
# 8/9 self-test (one 480p render on a synthetic image)
# --------------------------------------------------------------------------- #

Write-Step 'Self-test (one 480p render on a synthetic image)'
$code = Invoke-Native $python.Exe (@($python.Args) + @('-m', 'qwenstyle', 'selftest', '--profile', $Profile))
if ($code -ne 0) { Fail ('selftest failed with exit code ' + $code) $code }

# --------------------------------------------------------------------------- #
# 9/9 next steps
# --------------------------------------------------------------------------- #

Write-Step 'Ready'

Write-Host ''
Write-Host '    Put a scene folder together:' -ForegroundColor White
Write-Note 'promts\my-scene\prompt.txt   (FUSAL prompt, see PROMPT_RULES.md)'
Write-Note 'promts\my-scene\image1.png    (optional references, up to 10)'
Write-Host ''
Write-Host '    Generate:' -ForegroundColor White
Write-Note ($python.Exe + ' -m qwenstyle gen --preset ' + $Profile + ' --prompt my-scene --count 1')
Write-Host ''
Write-Host '    Presets of this profile:' -ForegroundColor White
$presets = Get-ProfilePresets $python $Profile
if ($presets) { Write-Note $presets } else { Write-Note 'python -m qwenstyle models list' }
Write-Host ''
Write-Host '    Docs: README.md (quick start), PROMPT_RULES.md (how to write a prompt)' -ForegroundColor White
Write-Host ''

exit 0
