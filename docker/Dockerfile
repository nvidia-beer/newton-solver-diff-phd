# syntax=docker/dockerfile:1
ARG UV_VERSION=0.13.6
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# Pin the same profiler build on both architectures. Extract on the build host;
# only the CLI installation (including its licenses) enters the runtime image.
FROM --platform=$BUILDPLATFORM ${CUDA_IMAGE} AS nsight
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$TARGETARCH" in \
        amd64) \
            package=NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb; \
            checksum=61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08 ;; \
        arm64) \
            package=nsight-systems-cli-2026.5.1_2026.5.1.161-1_arm64.deb; \
            checksum=087c08b5ddf3ed6f913748f8206d6bbdc225f2aa275ff33af4c487e8b319c8ab ;; \
        *) echo "Unsupported Nsight Systems architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl --fail --location --retry 3 \
        "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/$TARGETARCH/$package" \
        --output /tmp/nsight.deb; \
    printf '%s  %s\n' "$checksum" /tmp/nsight.deb | sha256sum --check -; \
    test "$(dpkg-deb -f /tmp/nsight.deb Architecture)" = "$TARGETARCH"; \
    test "$(dpkg-deb -f /tmp/nsight.deb Version)" = '2026.5.1.161-265138896106v0'; \
    dpkg-deb --extract /tmp/nsight.deb /tmp/nsight; \
    mv /tmp/nsight/opt/nvidia/nsight-systems-cli/2026.5.1 /opt/nsight-systems; \
    rm -rf /tmp/nsight /tmp/nsight.deb

FROM ${CUDA_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg git python3.12 python3.12-venv libpython3.12t64 libglib2.0-0t64 \
    libgl1 libegl1 libopengl0 libglu1-mesa libglfw3 \
    libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libxrender1 libxfixes3 libxkbcommon0 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=nsight /opt/nsight-systems /opt/nsight-systems
RUN /opt/nsight-systems/bin/nsys --version

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NSYS_PATH=/opt/nsight-systems/bin/nsys \
    PATH="/opt/venv/bin:/opt/nsight-systems/bin:${PATH}"

WORKDIR /workspace
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra examples --extra profiling --python /usr/bin/python3.12
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
