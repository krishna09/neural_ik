# Running Neural IK in WSL with the GPU

How this project is set up to run on this laptop (Windows 11, RTX 3060 Laptop GPU)
inside WSL2 Ubuntu 24.04, and how to use it day to day.

## Quick start

```powershell
wsl                 # opens Ubuntu-24.04 as root (the default distro and user)
```
```bash
nik                 # cd to the project + activate the GPU venv
nik-server          # start Neural IK Studio
```
Then open **http://127.0.0.1:8765** in a Windows browser. Press `Ctrl+C` to stop the server.

## Why WSL and not Windows Python

Two things block running this project with Windows Python on this machine:

| Problem | Effect |
|---|---|
| Only Python 3.14 is installed on Windows, and pybullet has no Windows wheel for it | `pip install pybullet` fails |
| Windows **Smart App Control** is on | torch's DLLs are blocked (`WinError 4551: An Application Control policy has blocked this file`) |

Smart App Control does not apply to Linux binaries inside WSL, so the project runs there.
The project files stay on the H: drive and are shared by Windows and WSL. Nothing is copied.

## What is installed

| Component | Details |
|---|---|
| WSL2 + Ubuntu 24.04 | Default distro (`wsl -l -v` shows `* Ubuntu-24.04`) |
| Default user | `root` (set in `/etc/wsl.conf` → `[user] default=root`) |
| Project path in WSL | `/mnt/h/STP-2026/Projects/vs_code_claude/neural_ik`, with a link at `/root/neural_ik` |
| Python | 3.12 (Ubuntu system Python) |
| GPU venv | `/root/nik-venv-gpu`: torch **2.11.0+cu128**, pybullet **3.2.7** (built from source), numpy, matplotlib, psutil |
| NVIDIA driver (Windows) | **617.14** (was 461.79, which is too old for CUDA in WSL) |

`/root/.bashrc` contains:
```bash
# >>> neural_ik >>>
export PATH="/usr/lib/wsl/lib:$PATH"   # nvidia-smi / CUDA from the Windows driver
nik() {   # go to the project and activate the GPU venv
    cd ~/neural_ik && source ~/nik-venv-gpu/bin/activate
}
alias nik-server='nik && python ui/server.py --no-browser'
# <<< neural_ik <<<
```

## The UI server

Neural IK Studio (`ui/server.py`) is **not Django**. It uses Python's built-in
`http.server` (`ThreadingHTTPServer`). The front end is plain HTML/JS with three.js in `ui/static/`.

```bash
nik
python ui/server.py --no-browser              # default http://127.0.0.1:8765
python ui/server.py --no-browser --port 9000  # another port
```

Every dataset and training run made from the UI is saved in its own folder under
`ui/workspace/library/`. Nothing is overwritten.

## Command-line pipeline

```bash
nik
python generate_dataset.py --robot xarm6 --n 20000 --seed 0
python train_ik.py --robot xarm6 --epochs 60 --collision-epochs 15
python evaluate.py --robot xarm6 --n 200
python make_figures.py
```
Robots: `xarm6`, `kuka_iiwa`, `panda`.

> **Warning:** `train_ik.py` overwrites `models/<robot>_ik_net.pt`, `_collision_net.pt` and
> `_history.json`, which are committed to git. Restore with `git checkout models/`.
> Training from the UI never overwrites anything.

## What runs on the GPU

| Part | Device | Notes |
|---|---|---|
| Training (UI and `train_ik.py`) | **GPU** | picks `cuda` automatically |
| Evaluate in the UI (batch eval + refinement) | **GPU** | |
| Run Inference in the UI (one target) | CPU | on purpose: for one sample the CPU is as fast |
| LM refinement in Run Inference | CPU | on purpose: never competes with a training job |
| Dataset generation | CPU | PyBullet is CPU-only |
| `evaluate.py` (standalone) | CPU | small script |

Check the GPU:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
A training run's `ui/workspace/library/runs/<run>/meta.json` records `"device"` and `"gpu"`.
The UI's System Monitor shows live GPU usage.

## VS Code in WSL

1. Install the **WSL** extension (`ms-vscode-remote.remote-wsl`).
2. `Ctrl+Shift+P` → **WSL: Open Folder in WSL…** → `/root/neural_ik`
   (or `/mnt/h/STP-2026/Projects/vs_code_claude/neural_ik`).
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → `/root/nik-venv-gpu/bin/python`.

The window title shows **[WSL: Ubuntu-24.04]** when connected.
Opening the folder from Windows (`H:\...`) uses Windows Python, which cannot run this project.

## Git line endings

Windows git uses `core.autocrlf=true`, so files on disk have CRLF line endings. Without the same
setting, WSL git shows ~116 files as modified. WSL git is set to match:
```bash
git config --global core.autocrlf true
```
`git status` is now clean on both sides.

## Claude Code chat history

Claude Code stores history per project folder, so WSL starts with none. This project's
history (from `C:\Users\acer\.claude\projects\h--STP-2026-Projects-vs-code-claude-neural-ik\`)
was copied to:

- `/root/.claude/projects/-mnt-h-STP-2026-Projects-vs-code-claude-neural-ik/`
- `/root/.claude/projects/-root-neural-ik/`

Open it from the Claude Code panel's past conversations, or `/resume` in the CLI.
The copy is a snapshot. Old messages refer to Windows paths (`h:\STP-2026\...`).
To copy newer Windows history again:
```bash
for d in -mnt-h-STP-2026-Projects-vs-code-claude-neural-ik -root-neural-ik; do
  mkdir -p ~/.claude/projects/$d
  cp -r /mnt/c/Users/acer/.claude/projects/h--STP-2026-Projects-vs-code-claude-neural-ik/. ~/.claude/projects/$d/
done
```

## Rebuild from scratch

1. **Windows:** update the NVIDIA driver (nvidia.com → GeForce RTX 30 Series (Notebooks) →
   RTX 3060 Laptop GPU), then restart.
2. **Windows (admin):** `wsl --install -d Ubuntu-24.04`, then restart.
3. **Ubuntu (root):**
   ```bash
   apt-get update && apt-get install -y python3-venv python3-pip build-essential
   python3 -m venv ~/nik-venv-gpu
   ~/nik-venv-gpu/bin/pip install --upgrade pip
   ~/nik-venv-gpu/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
   cd /mnt/h/STP-2026/Projects/vs_code_claude/neural_ik
   ~/nik-venv-gpu/bin/pip install -r requirements.txt psutil   # pybullet compiles: ~10-15 min
   ln -sfn /mnt/h/STP-2026/Projects/vs_code_claude/neural_ik ~/neural_ik
   git config --global core.autocrlf true
   ```
4. Add the `.bashrc` block above, and `[user]` / `default=root` to `/etc/wsl.conf`,
   then `wsl --terminate Ubuntu-24.04` from Windows.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | `export PATH=/usr/lib/wsl/lib:$PATH`; after a driver update run `wsl --shutdown` in Windows and reopen |
| `nvidia-smi: command not found` | same PATH line (it lives in `/usr/lib/wsl/lib`) |
| `nik: command not found` | logged in as another user: use `wsl -u root`, or check `/etc/wsl.conf` |
| `Port 8765 is already in use` | a server is already running: `pkill -f ui/server.py`, or use `--port` |
| `git status` shows ~116 modified files in WSL | `git config --global core.autocrlf true` |
| Browser can't reach `127.0.0.1:8765` | the WSL terminal running the server must stay open |

## Cleanup (optional)

These are no longer needed:

- `/root/nik-venv` (1.3 GB): old CPU-only venv. `rm -rf /root/nik-venv`
- `.venv/` in the project folder (749 MB, gitignored): Windows venv that Smart App Control blocks.
