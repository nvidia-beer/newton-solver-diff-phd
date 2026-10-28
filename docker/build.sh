#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project_name=$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["name"])' "$project_dir/pyproject.toml")
image=${SOLVER_IMAGE:-$project_name:local}
build_options=()

usage() {
    cat <<'EOF'
Usage: docker/build.sh [--image TAG] [--platform linux/amd64|linux/arm64] [-- BUILD_OPTIONS...]

Builds for the host architecture unless --platform is specified.
SOLVER_IMAGE overrides the default image tag (<project-name>:local).
EOF
}

while (($#)); do
    case "$1" in
        --image|--platform)
            if (($# < 2)); then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            if [[ $1 == --image ]]; then
                image=$2
            else
                build_options+=(--platform "$2")
            fi
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; build_options+=("$@"); break ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f $project_dir/pyproject.toml || ! -f $project_dir/uv.lock ]]; then
    printf 'Build requires pyproject.toml and uv.lock in %s\n' "$project_dir" >&2
    exit 1
fi

exec docker build --tag "$image" --file "$project_dir/docker/Dockerfile" "${build_options[@]}" "$project_dir"
