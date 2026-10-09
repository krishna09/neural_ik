# Run on the PC WITH internet (from the project folder):
#   powershell -ExecutionPolicy Bypass -File docker\export_image.ps1            # GPU image (CUDA 12.6)
#   powershell -ExecutionPolicy Bypass -File docker\export_image.ps1 -Variant cpu
# Builds the image and saves it to neuralik_image.tar - copy that file (and the project
# folder) to the offline PC, then run docker\load_and_run.ps1 / load_and_run.sh there.
param([string]$Variant = "cu126", [string]$Out = "neuralik_image.tar")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:TORCH_VARIANT = $Variant
docker compose build
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }

Write-Host "Saving image neuralik:latest -> $Out (this takes a few minutes)..."
docker save -o $Out neuralik:latest
if ($LASTEXITCODE -ne 0) { throw "docker save failed" }

$f = Get-Item $Out
$hash = (Get-FileHash $Out -Algorithm SHA256).Hash
"{0}  {1}" -f $hash, $f.Name | Out-File -Encoding ascii "$Out.sha256"
Write-Host ("Done: {0}  {1:N1} GB  SHA256 {2}" -f $f.FullName, ($f.Length / 1GB), $hash)
Write-Host "Copy $Out, $Out.sha256 and the project folder to the offline PC."
