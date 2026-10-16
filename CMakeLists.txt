cmake_minimum_required(VERSION 3.18)
project(reflex_cv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BUILD_TESTS "Build the host-side unit tests (no GPU required)" ON)

# ---------------------------------------------------------------------------
# Why CUDA is enabled conditionally rather than in the project() line
# ---------------------------------------------------------------------------
# project(... LANGUAGES CUDA) is a hard failure when no CUDA compiler is
# present, which would mean a teammate part-way through setting up their
# toolchain cannot even configure the project - and therefore cannot run the
# unit tests, which need none of this. Probing for CUDA instead means the tests
# always build and the GPU pipeline builds when the GPU toolchain is there.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # IMPORTANT: list every GPU architecture either of you builds on, so the
    # same source compiles on both machines. Check each one with
    #   nvidia-smi --query-gpu=compute_cap --format=csv
    # and do not guess.
    #   RTX 3060 Ti (Ampere)  -> 86
    #   RTX 4060/4070 (Ada)   -> 89
    #
    # This MUST be set before enable_language(CUDA), and as a CACHE variable.
    # enable_language() assigns its own default (often 52) if the variable is
    # not already set, so a plain `if(NOT DEFINED ...)` guard placed after it
    # never fires and the build silently targets the wrong architecture - it
    # still runs, via JIT recompilation at startup, just slower and with a
    # long first-launch pause. CACHE without FORCE means an explicit
    # -DCMAKE_CUDA_ARCHITECTURES=89 on the command line still wins.
    set(CMAKE_CUDA_ARCHITECTURES "86;89" CACHE STRING
        "GPU compute capabilities to generate code for")

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenCV QUIET)
find_package(CUDAToolkit QUIET)

# TensorRT ships no CMake config file, so it is located by hand. The HINTS
# cover a tarball install (TENSORRT_ROOT) and a .deb install, which put things
# in different places.
set(TENSORRT_ROOT "" CACHE PATH "Path to your local TensorRT installation")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include
          /usr/include/x86_64-linux-gnu
          /usr/local/tensorrt/include)
find_library(TENSORRT_LIB nvinfer
    HINTS ${TENSORRT_ROOT}/lib
          ${TENSORRT_ROOT}/lib64
          /usr/lib/x86_64-linux-gnu)

set(HAVE_GPU_STACK FALSE)
if(CMAKE_CUDA_COMPILER AND OpenCV_FOUND AND CUDAToolkit_FOUND)
    set(HAVE_GPU_STACK TRUE)
endif()

# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
if(HAVE_GPU_STACK AND TENSORRT_INCLUDE_DIR AND TENSORRT_LIB)
    add_executable(reflex_cv
        src/main.cpp
        src/preprocess.cu
        src/trt_infer.cpp
        src/postprocess.cpp
        src/command_shaper.cpp
        src/esp32_client.cpp
    )
    target_include_directories(reflex_cv PRIVATE
        include src ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(reflex_cv PRIVATE
        ${OpenCV_LIBS} CUDA::cudart ${TENSORRT_LIB})

    # Separable compilation is deliberately OFF. There is one .cu file and no
    # device function is called across translation units, so the extra device
    # link step buys nothing and only prevents the compiler from inlining the
    # geometry helpers into the kernel - which matters for a memory-bound
    # kernel launched thousands of times a minute.
    set_target_properties(reflex_cv PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    if(UNIX)
        target_link_libraries(reflex_cv PRIVATE pthread)
    endif()
    if(WIN32)
        target_link_libraries(reflex_cv PRIVATE ws2_32)
    endif()
    message(STATUS "reflex_cv: will build")
else()
    message(WARNING
        "reflex_cv (the pipeline) will NOT be built - something is missing:\n"
        "    CUDA compiler : ${CMAKE_CUDA_COMPILER}\n"
        "    CUDAToolkit   : ${CUDAToolkit_FOUND}\n"
        "    OpenCV        : ${OpenCV_FOUND}\n"
        "    TensorRT inc  : ${TENSORRT_INCLUDE_DIR}\n"
        "    TensorRT lib  : ${TENSORRT_LIB}\n"
        "  For TensorRT, pass -DTENSORRT_ROOT=/path/to/TensorRT-10.x\n"
        "  The unit tests below do not need any of this and will still build.")
endif()

# ---------------------------------------------------------------------------
# Kernel correctness tool (needs CUDA + OpenCV, but NOT TensorRT)
# ---------------------------------------------------------------------------
# Compares the CUDA preprocessing kernel against an OpenCV CPU implementation
# of the same transform. This is the step the build guide means by "verify
# correctness against OpenCV's CPU equivalent before trusting your kernel's
# output" - run it before wiring the kernel into the pipeline.
if(HAVE_GPU_STACK)
    add_executable(verify_preprocess
        tools/verify_preprocess.cpp
        src/preprocess.cu
    )
    target_include_directories(verify_preprocess PRIVATE
        include src ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(verify_preprocess PRIVATE
        ${OpenCV_LIBS} CUDA::cudart)
    set_target_properties(verify_preprocess PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "verify_preprocess: will build")
endif()

# ---------------------------------------------------------------------------
# Host-side unit tests - no GPU, no TensorRT, no OpenCV
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_preprocess_math test/test_preprocess_math.cpp)
    target_include_directories(test_preprocess_math PRIVATE src)

    add_executable(test_postprocess test/test_postprocess.cpp src/postprocess.cpp)
    target_include_directories(test_postprocess PRIVATE src)

    add_executable(test_command_shaper
        test/test_command_shaper.cpp src/command_shaper.cpp)
    target_include_directories(test_command_shaper PRIVATE src)

    # The Arduino firmware's logic, tested on the host. mcu/balance_core.h is
    # written with no Arduino dependency precisely so this is possible.
    add_executable(test_balance_core test/test_balance_core.cpp)
    target_include_directories(test_balance_core PRIVATE mcu)

    foreach(t test_preprocess_math test_postprocess test_command_shaper
              test_balance_core)
        add_test(NAME ${t} COMMAND ${t})
    endforeach()

    message(STATUS "tests: will build (run them with ctest)")
endif()

message(STATUS "")
message(STATUS "---- reflex-cv configuration ----")
message(STATUS "  build type   ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch    ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenCV       ${OpenCV_VERSION}")
message(STATUS "  TensorRT     ${TENSORRT_LIB}")
message(STATUS "---------------------------------")
