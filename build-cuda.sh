#!/usr/bin/env bash

set -Eeuo pipefail

IMAGE="triton-2-cuda:latest"
PLATFORM="linux/amd64"
BUILD_DIR="build"

usage() {
    cat <<EOF
Usage: $0 [--clean]

Build the CUDA project inside the development container.

Options:
  --clean    Remove the existing CMake build directory before compiling
  -h, --help Show this help message
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

trap 'echo "ERROR: build-cuda.sh failed at line $LINENO." >&2' ERR

CLEAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "Unknown option: $1"
            ;;
    esac
done

# Make sure Docker is available.
command -v docker >/dev/null 2>&1 \
    || die "Docker is not installed or is not available on PATH."

docker info >/dev/null 2>&1 \
    || die "Docker daemon is not running. Start Docker Desktop and try again."

# Make sure we're running from the repository root.
[[ -f "Dockerfile" ]] \
    || die "Dockerfile not found. Run this script from the repository root."

[[ -f "CMakeLists.txt" ]] \
    || die "CMakeLists.txt not found. Run this script from the repository root."

# Build the development image.
echo "==> Building Docker image: $IMAGE"
docker build \
    --platform "$PLATFORM" \
    -t "$IMAGE" \
    .

# Verify the image architecture.
IMAGE_PLATFORM="$(docker image inspect "$IMAGE" \
    --format '{{.Os}}/{{.Architecture}}')"

if [[ "$IMAGE_PLATFORM" != "$PLATFORM" ]]; then
    die "Docker image has platform $IMAGE_PLATFORM, expected $PLATFORM."
fi

echo "==> Using image: $IMAGE ($IMAGE_PLATFORM)"

# Optionally clean the CMake build directory.
if [[ "$CLEAN" == true ]]; then
    echo "==> Removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Run CMake + build inside the container.
echo "==> Configuring and compiling CUDA project"

docker run --rm -it \
    --platform "$PLATFORM" \
    -v "$PWD:/workspace" \
    -w /workspace \
    "$IMAGE" \
    bash -Eeuo pipefail -c '
        echo "==> Toolchain"
        echo "CUDA:"
        nvcc --version | tail -n 4

        echo
        echo "CMake:"
        cmake --version | head -n 1

        echo
        echo "Compiler:"
        g++ --version | head -n 1

        echo
        echo "==> CMake configure"
        cmake -S . -B build -G Ninja \
            -DCMAKE_CUDA_ARCHITECTURES=75

        echo
        echo "==> CMake build"
        cmake --build build --parallel

        echo
        echo "==> Build successful"
    '

echo
echo "========================================"
echo " CUDA build completed successfully"
echo "========================================"