# syntax=docker/dockerfile:1
ARG UV_VERSION=0.13.6
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ${CUDA_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg python3.12 python3.12-venv libpython3.12t64 \
    libgl1 libegl1 libopengl0 libglu1-mesa libglfw3 \
    libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libxrender1 libxfixes3 libxkbcommon0 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra examples --python /usr/bin/python3.12
# Check native runtime dependencies before publishing a usable image.
RUN python -c "import ctypes, importlib.util; ctypes.CDLL('libpython3.12.so.1.0'); \
    importlib.import_module('pxr.Usd').Stage.CreateInMemory() if importlib.util.find_spec('pxr') else None"
RUN mkdir -p /opt/solver-metadata \
    && cp pyproject.toml uv.lock /opt/solver-metadata/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/solver-entrypoint

# The editable install reads source from /workspace. Keep the built environment
# read-only for the host UID; dependency changes require rebuilding the image.
ENV UV_NO_SYNC=1 \
    XDG_CACHE_HOME=/cache \
    UV_CACHE_DIR=/cache/uv \
    WARP_CACHE_PATH=/cache/warp

ENTRYPOINT ["solver-entrypoint"]
CMD []
