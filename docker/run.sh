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
profile=0
profile_mounts=()
examples_main=$project_dir/src/$package/examples/__main__.py

usage() {
    cat <<'EOF'
Usage: docker/run.sh [OPTIONS] [EXAMPLE [EXAMPLE_OPTIONS...]]
       docker/run.sh [OPTIONS] -- COMMAND [ARGUMENTS...]
       docker/run.sh [OPTIONS] --nsys EXAMPLE [PROFILE_OPTIONS...]

Options:
  --list          List discovered examples without starting Docker.
  --nsys          Profile a README video example; --nsys --help lists options.
  --cpu           Run without an NVIDIA GPU; select the CPU example device.
  --headless      Select the null viewer; do not mount the X11 socket.
  --image TAG     Image to run (SOLVER_IMAGE or <project-name>:local).
  --platform P   Select the container platform; normally use the host default.

With no example, opens an interactive menu. Examples are discovered from example_*.py.
Visual examples use CUDA and X11 by default. Terminal-only examples need no display.
Profiling defaults to the null viewer. Use --nsys EXAMPLE --viewer gl for rendering.
NSYS_ROOT selects a host Nsight Systems installation (containing bin/nsys).
Standard /opt/nvidia and CUDA installations are discovered automatically and
mounted read-only only during profiling. Reports persist under nsys/results/.
EOF
}

while (($#)); do
    case "$1" in
        --nsys) profile=1; shift; break ;;
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
if ((profile)); then
    if (($# == 0)); then
        exec python3 "$project_dir/nsys/profile_examples.py" --help
    fi
    # Read-only commands require only Python, even before the Docker image exists.
    for argument in "$@"; do
        [[ $argument != -- ]] || break
        case $argument in
            --list|--help|-h|--dry-run)
                exec python3 "$project_dir/nsys/profile_examples.py" "$@"
                ;;
        esac
    done
    if ((!use_gpu)); then
        printf 'Nsight CUDA profiling requires a GPU; omit --cpu.\n' >&2
        exit 2
    fi
    # A host profiler cannot run under an emulated container architecture.
    case $(uname -m) in
        aarch64) host_platform=linux/arm64; nsys_arch=sbsa-armv8 ;;
        x86_64) host_platform=linux/amd64; nsys_arch=x64 ;;
        *) printf 'Unsupported Nsight host architecture.\n' >&2; exit 2 ;;
    esac
    if ((${#platform_options[@]})) && [[ ${platform_options[1]} != "$host_platform" ]]; then
        printf 'Nsight profiling requires the host platform %s.\n' "$host_platform" >&2
        exit 2
    fi
    nsys_root=${NSYS_ROOT:-}
    if [[ -z $nsys_root ]]; then
        candidates=()
        host_nsys=$(command -v nsys || true)
        if [[ -n $host_nsys ]]; then
            resolved=$(readlink -f -- "$host_nsys")
            candidates+=("$(dirname -- "$(dirname -- "$resolved")")")
        fi
        shopt -s nullglob
        candidates+=(/opt/nvidia/nsight-systems/* /opt/nvidia/nsight-systems-cli/*
            /usr/local/cuda/nsight-systems-* /usr/local/cuda-*/nsight-systems-*)
        shopt -u nullglob
        for candidate in "${candidates[@]}"; do
            if [[ -x $candidate/bin/nsys && -x $candidate/target-linux-$nsys_arch/nsys ]]; then
                nsys_root=$candidate
                break
            fi
        done
    fi
    if [[ -z $nsys_root || ! -x $nsys_root/bin/nsys ]]; then
        printf 'Install Nsight Systems on the host and set NSYS_ROOT to its installation directory (containing bin/nsys).\n' >&2
        exit 1
    fi
    nsys_root=$(cd -- "$nsys_root" && pwd)
    profile_mounts=(--mount "type=bind,src=$nsys_root,dst=/opt/nsight-systems,readonly"
        --env NSYS_PATH=/opt/nsight-systems/bin/nsys)
    viewer=null
    remaining=("$@")
    for ((i=0; i<${#remaining[@]}; i++)); do
        [[ ${remaining[i]} != -- ]] || break
        case ${remaining[i]} in
            --viewer) viewer=${remaining[i+1]:-}; ((i+=1)) ;;
            --viewer=*) viewer=${remaining[i]#--viewer=} ;;
        esac
    done
    headless=1
    [[ $viewer != gl ]] || headless=0
    command=(python /workspace/nsys/profile_examples.py --device "$device" "$@")
elif ((raw_command)); then
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
    "${platform_options[@]}" "${profile_mounts[@]}")
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
