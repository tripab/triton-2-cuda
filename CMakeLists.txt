cmake_minimum_required(VERSION 3.24)

# Build for the GPU(s) in this machine unless told otherwise, e.g.
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
# ("native" needs a visible GPU at configure time.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

project(triton-2-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Shared helpers (error checking, benchmarking) used by every tutorial.
add_library(common INTERFACE)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_compile_options(common INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

enable_testing()

# add_tutorial(<target> <sources>...)
#   Builds one tutorial executable and registers a quick correctness test
#   (`ctest`) that runs it with --test-only.
function(add_tutorial target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE common)
  add_test(NAME ${target} COMMAND ${target} --test-only)
endfunction()

# One subdirectory per Triton tutorial.
add_subdirectory(01-vector-add)