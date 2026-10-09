# Neural IK Studio - self-contained image for offline PCs.
#
# Build on a PC WITH internet, then copy the image to the offline PC (see DOCKER.md):
#   docker compose build                      (GPU build, PyTorch for CUDA 12.6)
#   docker compose build --build-arg TORCH_VARIANT=cpu      (smaller, CPU only)
#
# Python 3.13 = the version the project was developed with. pybullet 3.2.7 has no
# prebuilt Linux wheel for Python 3.13, so it is compiled once in the "wheels" stage;
# the final image contains no compiler.

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- stage 1: wheels
FROM python:${PYTHON_VERSION}-slim AS wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.lock /tmp/requirements.lock
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.lock

# ---------------------------------------------------------------- stage 2: runtime
FROM python:${PYTHON_VERSION}-slim

# cu126 = CUDA 12.6 build (runs on most NVIDIA drivers, falls back to CPU when there is no GPU)
# cu130 = CUDA 13.0 build (needs a newer driver)   cpu = CPU-only build (much smaller image)
ARG TORCH_VARIANT=cu126
ARG TORCH_VERSION=2.14.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch

RUN pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"
COPY --from=wheels /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels

WORKDIR /app
COPY . /app

# Non-root user (UID 1000 = the usual first user on Linux hosts). The library folder
# ui/workspace is the only place the app writes to; mount it as a volume to keep it.
RUN useradd --uid 1000 --create-home appuser \
 && mkdir -p /app/ui/workspace \
 && chown -R appuser:appuser /app/ui/workspace
USER appuser

EXPOSE 8765
VOLUME ["/app/ui/workspace"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/robots', timeout=4)" || exit 1

CMD ["python", "ui/server.py", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
