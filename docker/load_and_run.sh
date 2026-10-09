#!/usr/bin/env sh
# Run on the OFFLINE PC (Linux) from the project folder:   sh docker/load_and_run.sh
# Loads neuralik_image.tar (if the image is not loaded yet) and starts Neural IK Studio.
# Uses the NVIDIA GPU when nvidia-smi works on this PC, otherwise runs on the CPU.
set -e
cd "$(dirname "$0")/.."
IMAGE="${1:-neuralik_image.tar}"

if ! docker image inspect neuralik:latest >/dev/null 2>&1; then
  [ -f "$IMAGE" ] || { echo "Image not loaded and $IMAGE not found in $(pwd)"; exit 1; }
  if [ -f "$IMAGE.sha256" ]; then
    want=$(cut -d' ' -f1 "$IMAGE.sha256" | tr 'A-F' 'a-f')
    have=$(sha256sum "$IMAGE" | cut -d' ' -f1)
    [ "$want" = "$have" ] || { echo "$IMAGE is damaged (SHA256 mismatch) - copy it again"; exit 1; }
  fi
  echo "Loading $IMAGE ..."
  docker load -i "$IMAGE"
fi

mkdir -p ui/workspace
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"      # files in ui/workspace stay yours
if nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU found - starting with GPU."
  docker compose -f compose.yaml -f compose.gpu.yaml up -d
else
  echo "No NVIDIA GPU found - starting on CPU."
  docker compose -f compose.yaml up -d
fi
echo "Neural IK Studio is starting: open http://localhost:8765"
