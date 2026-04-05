$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
	Write-Host "Creating .venv and installing requirements..."
	python -m venv .venv
	& (Join-Path $PSScriptRoot ".venv\Scripts\pip.exe") install -r requirements.txt
}
& $py app.py
