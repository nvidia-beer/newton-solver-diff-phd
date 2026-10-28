#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
project_name=$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["name"])' "$project_dir/pyproject.toml")
package=${project_name//-/_}
image=${SOLVER_IMAGE:-$project_name:local}
use_gpu=1
headless=0
platform_options=()
raw_command=0
examples_main=$project_dir/src/$package/examples/__main__.py

usage() {
    cat <<'EOF'
Usage: docker/run.sh [OPTIONS] [EXAMPLE [EXAMPLE_OPTIONS...]]
       docker/run.sh [OPTIONS] -- COMMAND [ARGUMENTS...]

Options:
  --list          List discovered examples without starting Docker.
  --cpu           Run without an NVIDIA GPU; select the CPU example device.
  --headless      Select the null viewer; do not mount the X11 socket.
  --image TAG     Image to run (SOLVER_IMAGE or <project-name>:local).
  --platform P   Select the container platform; normally use the host default.

With no example, opens an interactive menu. Examples are discovered from example_*.py.
Visual examples use CUDA and X11 by default. Terminal-only examples need no display.
EOF
}

while (($#)); do
    case "$1" in
        --cpu) use_gpu=0; shift ;;
        --headless) headless=1; shift ;;
        --image|--platform)
            if (($# < 2)); then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            if [[ $1 == --image ]]; then
                image=$2
            else
                platform_options=(--platform "$2")
            fi
            shift 2
            ;;
        --list) python3 "$examples_main" --list; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        --) raw_command=1; shift; break ;;
        -*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        *) break ;;
    esac
done

device=cpu
((use_gpu)) && device=cuda:0
if ((raw_command)); then
    if (($# == 0)); then
        printf 'Expected a command after --.\n' >&2
        exit 2
    fi
    command=("$@")
else
    selection_args=(--select)
    if (($#)); then
        selection_args+=("$1")
        shift
    fi
    selection=$(python3 "$examples_main" "${selection_args[@]}")
    [[ -n $selection ]] || exit 0
    read -r example mode <<< "$selection"
    command=(python -m "$package.examples" "$example" --device "$device")
    if [[ $mode == headless ]]; then
        headless=1
    else
        viewer=gl
        ((headless)) && viewer=null
        # Match the example parser's last viewer option before deciding on X11.
        remaining=("$@")
        for ((i=0; i<${#remaining[@]}; i++)); do
            case ${remaining[i]} in
                --viewer) viewer=${remaining[i+1]:-}; ((i+=1)) ;;
                --viewer=*) viewer=${remaining[i]#--viewer=} ;;
            esac
        done
        headless=0
        [[ $viewer == gl ]] || headless=1
        for argument in "${remaining[@]}"; do
            [[ $argument != --help && $argument != -h && $argument != --list-assets ]] || headless=1
        done
        command+=(--viewer "$viewer")
    fi
    command+=("$@")
fi

cache_dir=$project_dir/.cache
mkdir -p "$cache_dir/uv" "$cache_dir/warp"
docker_options=(run --rm --init --user "$(id -u):$(id -g)" --workdir /workspace
    --mount "type=bind,src=$project_dir,dst=/workspace"
    --mount "type=bind,src=$cache_dir,dst=/cache"
    "${platform_options[@]}")
if [[ -t 0 && -t 1 ]]; then
    docker_options+=(-it)
fi

if ((use_gpu)); then
    capabilities=compute,utility
    ((headless)) || capabilities+=,graphics,display
    docker_options+=(--gpus all --env "NVIDIA_DRIVER_CAPABILITIES=$capabilities")
else
    docker_options+=(--env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void)
fi

if ((!headless)); then
    if [[ ${DISPLAY:-} != :* && ${DISPLAY:-} != unix:* ]]; then
        printf 'A local X11 DISPLAY is required. Use --headless for a terminal-only run.\n' >&2
        exit 1
    fi
    xauthority=${XAUTHORITY:-${HOME}/.Xauthority}
    if [[ ! -d /tmp/.X11-unix || ! -f $xauthority || ! -r $xauthority ]]; then
        printf 'Set XAUTHORITY to your readable X11 authorization file, or use --headless.\n' >&2
        exit 1
    fi
    # Preserve FamilyLocal cookie hostname matching without changing X access.
    docker_options+=(--hostname "$(hostname)" --env "DISPLAY=$DISPLAY" --env XAUTHORITY=/tmp/solver.Xauthority
        --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly
        --mount "type=bind,src=$xauthority,dst=/tmp/solver.Xauthority,readonly")
fi

exec docker "${docker_options[@]}" "$image" "${command[@]}"
