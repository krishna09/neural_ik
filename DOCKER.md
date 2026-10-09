# Running Neural IK Studio with Docker (online → offline PC)

An offline PC cannot download Python packages, PyTorch or the base image. So the **image is built
on the PC with internet** and copied to the offline PC as one file. The code itself comes along
through Git (or a USB copy) for the compose files and scripts; everything the app needs to run is
already inside the image.

```
PC with internet                         USB / network share                 Offline PC
───────────────                          ───────────────────                 ──────────
git push / copy project      ──────►     project folder           ──────►   git clone / copy
docker\export_image.ps1      ──────►     neuralik_image.tar (+.sha256) ──►   docker\load_and_run.ps1 (Windows)
                                                                             sh docker/load_and_run.sh (Linux)
                                                                             open http://localhost:8765
```

## 1. On the PC with internet

Requirements: Docker Desktop (Windows) or Docker Engine (Linux).

```powershell
cd <project folder>
powershell -ExecutionPolicy Bypass -File docker\export_image.ps1            # GPU image (default)
# or, for a much smaller CPU-only image:
powershell -ExecutionPolicy Bypass -File docker\export_image.ps1 -Variant cpu
```

This builds `neuralik:latest` and writes **`neuralik_image.tar`** plus `neuralik_image.tar.sha256`
into the project folder. The `.tar` files are ignored by Git, so never commit them; copy them with a USB
drive or a network share.

| Variant | PyTorch | Use when the offline PC… |
|---|---|---|
| `cu126` (default) | CUDA 12.6 | has an NVIDIA GPU (driver ≥ 525). Also runs on CPU if there is no GPU. |
| `cu130` | CUDA 13.0 | has an NVIDIA GPU with a very new driver (≥ 580) |
| `cpu` | CPU only | has no NVIDIA GPU – much smaller image |

What the build does: Python 3.13 (the version the project was developed with), the exact package
versions in `docker/requirements.lock`, and PyBullet 3.2.7 compiled from source (PyBullet has no
ready-made Linux package for Python 3.13). The 3D view's three.js library is bundled in
`ui/static/vendor/three/`, so the browser needs no internet either.

## 2. Copy to the offline PC

- the project folder (`git clone` from your Git server / a `git bundle`, or a plain copy)
- `neuralik_image.tar` and `neuralik_image.tar.sha256` → put both in the project folder
- optional: your library of datasets and trained runs, `ui/workspace/` – copy the whole folder into
  the same place in the project folder on the offline PC (it is not in Git)

## 3. On the offline PC

Requirements: Docker Desktop (Windows) or Docker Engine + Compose plugin (Linux). For the GPU:
NVIDIA driver (+ NVIDIA Container Toolkit on Linux; Docker Desktop on Windows/WSL2 has GPU support built in).

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File docker\load_and_run.ps1
```
```sh
# Linux
sh docker/load_and_run.sh
```

The script checks the file is complete (SHA256), loads the image once, then starts the container –
with the GPU if `nvidia-smi` works, otherwise on the CPU. Open **http://localhost:8765**.

## Everyday commands (run in the project folder)

| What | Command |
|---|---|
| Start (CPU) | `docker compose up -d` |
| Start (GPU) | `docker compose -f compose.yaml -f compose.gpu.yaml up -d` |
| Stop | `docker compose down` |
| Live log | `docker compose logs -f` |
| Is the GPU used? | `docker compose exec neuralik python -c "import torch; print(torch.cuda.is_available())"` |
| Other port (e.g. 9000) | `NEURALIK_PORT=9000 docker compose up -d` (PowerShell: `$env:NEURALIK_PORT=9000; docker compose up -d`) |

The library (datasets, trained runs, renders) is stored on the host in **`ui/workspace/`**, so it
survives stopping, removing or updating the container.

## Updating the offline PC later

Change code on the online PC → run `docker\export_image.ps1` again → copy the new `.tar` → on the
offline PC: `docker compose down`, `docker load -i neuralik_image.tar`, start again. The library in
`ui/workspace/` is kept.

## Troubleshooting

| Problem | Fix |
|---|---|
| `could not select device driver "nvidia"` | No GPU support on that PC – start without `compose.gpu.yaml` (CPU). |
| `pull access denied for neuralik` / Compose tries to build | The image is not loaded yet: `docker load -i neuralik_image.tar`. |
| `PermissionError` writing `ui/workspace` (Linux) | Start with your user id: `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d` (the Linux script does this). |
| `port is already allocated` | Another program uses 8765 – set `NEURALIK_PORT` (see above). |
| Page loads but no 3D robot | Hard-refresh the browser (Ctrl+F5) once after updating. |
