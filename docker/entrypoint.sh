#!/usr/bin/env bash
set -euo pipefail

for metadata in pyproject.toml uv.lock; do
    if ! cmp -s "/workspace/$metadata" "/opt/solver-metadata/$metadata"; then
        printf 'Container dependency metadata differs: %s. Rebuild with ./docker/build.sh.\n' "$metadata" >&2
        exit 1
    fi
done

if (($# == 0)); then
    project_name=$(python -c 'import tomllib; print(tomllib.load(open("/workspace/pyproject.toml", "rb"))["project"]["name"])')
    package=${project_name//-/_}
    set -- python -m "$package.examples"
fi

exec "$@"
