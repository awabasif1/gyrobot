#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sstream>

// Every CUDA runtime call in this project is wrapped in this macro.
// Silent CUDA failures are the #1 cause of "works on my machine" bugs in
// pipelines that also do OpenCV/TensorRT work on the same stream, because
// an error from stage N often only *manifests* as garbage output in stage N+1.
// Failing loudly, at the exact call site, is worth the verbosity.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::ostringstream oss;                                           \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                << " in call '" << #call << "': "                             \
                << cudaGetErrorString(err__);                                 \
            throw std::runtime_error(oss.str());                              \
        }                                                                     \
    } while (0)

// For kernel launches specifically: cudaGetLastError() catches launch-config
// errors (bad grid/block dims, too much shared mem requested, etc.) that a
// bare kernel<<<>>>() call would otherwise swallow.
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
    } while (0)
