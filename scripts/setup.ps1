$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Push-Location frontend
try {
 npm ci
 if ($LASTEXITCODE) { exit $LASTEXITCODE }
 npm run build
 if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally { Pop-Location }
Write-Host 'Готово. Запустите scripts/start.ps1'
