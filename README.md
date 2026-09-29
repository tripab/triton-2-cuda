# Triton tutorials in CUDA C++

CUDA C++ versions of the kernels in the
[OpenAI Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html),
written for practice. Each tutorial lives in its own directory and builds into one
executable. The executable checks the kernel against a CPU reference and then
runs a benchmark that mirrors the one in the Triton tutorial.

## Requirements

- NVIDIA GPU and driver
- CUDA Toolkit 12.x (`nvcc`)
- CMake 3.24 or newer and a C++17 host compiler

## Build and run

```bash
cmake -B build                                # builds for the GPU in this machine
# or pick the architecture yourself:  cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

ctest --test-dir build --output-on-failure    # correctness checks only
./build/01-vector-add/vector_add              # correctness check + benchmark
```

Every tutorial binary accepts `--test-only` to skip the benchmark.

## Layout

```
common/
  cuda_utils.cuh   CUDA_CHECK, cdiv, DeviceBuffer, device info
  bench.cuh        benchmark(): the C++ counterpart of triton.testing.do_bench
NN-name/
  CMakeLists.txt   add_tutorial(<target> <sources>)
  *.cu             kernels, host launcher, test, and benchmark
```

To add a tutorial, create `NN-name/`, call `add_tutorial(...)` in its
`CMakeLists.txt`, and add `add_subdirectory(NN-name)` to the root `CMakeLists.txt`.

## Tutorials

| # | Triton tutorial | CUDA version |
|---|-----------------|--------------|
| 01 | [Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | [`01-vector-add/vector_add.cu`](01-vector-add/vector_add.cu) |

## Profiling

```bash
ncu --set full -o vadd ./build/01-vector-add/vector_add --test-only
nsys profile ./build/01-vector-add/vector_add
```