# Run on the OFFLINE PC (Windows, Docker Desktop) from the project folder:
#   powershell -ExecutionPolicy Bypass -File docker\load_and_run.ps1
# Loads neuralik_image.tar (if the image is not loaded yet) and starts Neural IK Studio.
# Uses the NVIDIA GPU when nvidia-smi works on this PC, otherwise runs on the CPU.
param([string]$Image = "neuralik_image.tar")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

docker image inspect neuralik:latest *> $null
if ($LASTEXITCODE -ne 0) {
    if (-not (Test-Path $Image)) { throw "Image not loaded and $Image not found in $(Get-Location)" }
    if (Test-Path "$Image.sha256") {
        $want = (Get-Content "$Image.sha256").Split(" ")[0]
        $have = (Get-FileHash $Image -Algorithm SHA256).Hash
        if ($want -ne $have) { throw "$Image is damaged (SHA256 mismatch) - copy it again" }
    }
    Write-Host "Loading $Image ..."
    docker load -i $Image
    if ($LASTEXITCODE -ne 0) { throw "docker load failed" }
}

New-Item -ItemType Directory -Force ui\workspace | Out-Null
$files = @("-f", "compose.yaml")
$gpu = $false
try { nvidia-smi *> $null; $gpu = ($LASTEXITCODE -eq 0) } catch { $gpu = $false }
if ($gpu) { $files += @("-f", "compose.gpu.yaml"); Write-Host "NVIDIA GPU found - starting with GPU." }
else { Write-Host "No NVIDIA GPU found - starting on CPU." }

docker compose @files up -d
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
Write-Host "Neural IK Studio is starting: open http://localhost:8765"
