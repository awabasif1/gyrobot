cmake_minimum_required(VERSION 3.18)
project(reflex_cv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# IMPORTANT: list every GPU architecture either of you is building on, so the
# same source compiles correctly on both machines. Update these numbers to
# match your actual GPUs (check with `nvidia-smi --query-gpu=compute_cap
# --format=csv` on each machine) - do not just guess.
#   RTX 3060 Ti (Ampere)      -> 86
#   Add your friend's GPU's compute capability here once confirmed, e.g. 89
set(CMAKE_CUDA_ARCHITECTURES 86 89)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT has no official CMake find-module, so it's located manually.
# Adjust TENSORRT_ROOT if your install path differs (common on Windows vs Linux).
set(TENSORRT_ROOT "" CACHE PATH "Path to your local TensorRT installation")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT}/include)
find_library(TENSORRT_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib)

add_executable(reflex_cv
    src/main.cpp
    src/preprocess.cu
    src/trt_infer.cpp
)

target_include_directories(reflex_cv PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(reflex_cv PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TENSORRT_LIB}
)

set_target_properties(reflex_cv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
