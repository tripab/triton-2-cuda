#!/usr/bin/env bash

set -Eeuo pipefail

IMAGE="triton-cuda-dev:latest"
PLATFORM="linux/amd64"
BUILD_DIR="build"

CLEAN=false
TEST=false

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build the CUDA project inside the development container.

Options:
  --clean        Remove the existing CMake build directory first
  --test         Run CTest after a successful build
                 Requires an NVIDIA GPU and NVIDIA Container Toolkit
  -h, --help     Show this help message

Examples:
  $0
      Configure and compile.

  $0 --clean
      Clean, configure and compile.

  $0 --test
      Configure, compile and run correctness tests.
      Requires an NVIDIA GPU.

  $0 --clean --test
      Clean, compile and run correctness tests.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

trap 'echo "ERROR: build-cuda.sh failed at line $LINENO." >&2' ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            TEST=true
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

# ---------------------------------------------------------------------------
# Host-side validation
# ---------------------------------------------------------------------------

command -v docker >/dev/null 2>&1 \
    || die "Docker is not installed or is not available on PATH."

docker info >/dev/null 2>&1 \
    || die "Docker daemon is not running. Start Docker Desktop and try again."

[[ -f "Dockerfile" ]] \
    || die "Dockerfile not found. Run this script from the repository root."

[[ -f "CMakeLists.txt" ]] \
    || die "CMakeLists.txt not found. Run this script from the repository root."

# ---------------------------------------------------------------------------
# Build development image
# ---------------------------------------------------------------------------

echo "==> Building Docker image: $IMAGE"

docker build \
    --platform "$PLATFORM" \
    -t "$IMAGE" \
    .

IMAGE_PLATFORM="$(docker image inspect "$IMAGE" \
    --format '{{.Os}}/{{.Architecture}}')"

[[ "$IMAGE_PLATFORM" == "$PLATFORM" ]] \
    || die "Docker image has platform $IMAGE_PLATFORM, expected $PLATFORM."

echo "==> Using image: $IMAGE ($IMAGE_PLATFORM)"

# ---------------------------------------------------------------------------
# Optional clean
# ---------------------------------------------------------------------------

if [[ "$CLEAN" == true ]]; then
    echo "==> Removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# ---------------------------------------------------------------------------
# Configure Docker GPU mode
# ---------------------------------------------------------------------------

if [[ "$TEST" == true ]]; then
    echo "==> GPU test mode enabled"
    echo "    NVIDIA GPU + NVIDIA Container Toolkit required"
else
    echo "==> Compile-only mode (no GPU required)"
fi

# ---------------------------------------------------------------------------
# Run container
# ---------------------------------------------------------------------------

echo "==> Running CUDA build"

if [[ "$TEST" == true ]]; then

    docker run --rm \
        --platform "$PLATFORM" \
        --gpus all \
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
            echo "==> GPU"
            nvidia-smi --query-gpu=name,compute_cap,driver_version \
                --format=csv,noheader

            echo
            echo "==> CMake configure"
            cmake -B build

            echo
            echo "==> CMake build"
            cmake --build build -j

            echo
            echo "==> Running correctness tests"
            ctest --test-dir build --output-on-failure

            echo
            echo "==> Correctness tests passed"
        '

else

    docker run --rm \
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
            cmake -B build

            echo
            echo "==> CMake build"
            cmake --build build -j
        '

fi

echo
echo "========================================"

if [[ "$TEST" == true ]]; then
    echo " CUDA build + correctness tests passed"
else
    echo " CUDA build completed successfully"
fi

echo "========================================"