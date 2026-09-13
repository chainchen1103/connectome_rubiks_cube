# Launch from any directory, retaining relative output paths inside this checkout.
$ErrorActionPreference = 'Stop'
$repoDirectory = Split-Path -Parent $PSScriptRoot
$cliArguments = @($args)
if ($cliArguments.Count -eq 0) {
    $cliArguments = @('serve', '--open')
}

$pythonCandidates = @()
$venvPython = Join-Path $repoDirectory '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $pythonCandidates += $venvPython
}
$pathPython = Get-Command python -ErrorAction SilentlyContinue
if ($pathPython) {
    $pythonCandidates += $pathPython.Source
}
if ($env:USERPROFILE) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (Test-Path -LiteralPath $bundledPython) {
        $pythonCandidates += $bundledPython
    }
}

$selectedPython = $null
foreach ($candidatePython in ($pythonCandidates | Select-Object -Unique)) {
    # Do not select a Windows Store alias or an environment missing NumPy.
    try {
        & $candidatePython -c 'import sys, numpy; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $selectedPython = $candidatePython
            break
        }
    } catch {
        continue
    }
}
if (-not $selectedPython) {
    throw 'Python 3.10+ with NumPy was not found. In the repository run: py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e .'
}

Push-Location -LiteralPath $repoDirectory
try {
    Write-Host "Python: $selectedPython"
    & $selectedPython -m connectome_lab @cliArguments
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
