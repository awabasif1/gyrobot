#pragma once
// ===========================================================================
// COMPILE-CHECK STUB - NOT TensorRT, NOT USED BY THE REAL BUILD
// ===========================================================================
//
// The real NvInfer.h ships with the NVIDIA TensorRT SDK, which cannot be
// redistributed in a git repo. This file declares only the handful of TensorRT
// symbols src/trt_infer.cpp actually touches, with the signatures the real SDK
// documents, so that:
//
//   * `make check` can type-check trt_infer.cpp and main.cpp on a machine with
//     no TensorRT installed (CI, a teammate's laptop mid-setup, a code review),
//   * and the set of API calls this project depends on is written down in one
//     short, readable place.
//
// WHAT THIS DOES AND DOES NOT PROVE. Compiling against this stub proves the
// code is internally consistent and calls these functions with the right
// argument counts and types. It does NOT prove conformance with the real SDK:
// if a signature here is wrong, the stub build passes and the real build
// fails. The real build is the authority. Treat a green stub build as "no
// typos", not as "TensorRT integration verified".
//
// CMakeLists.txt never adds this directory to the real target's include path -
// only the `trt_stub_check` target does.
// ===========================================================================

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 0
#define NV_TENSORRT_PATCH 0

namespace nvinfer1 {

using AsciiChar = char;

enum class DataType : int32_t {
    kFLOAT = 0,
    kHALF  = 1,
    kINT8  = 2,
    kINT32 = 3,
    kBOOL  = 4,
    kUINT8 = 5,
};

enum class TensorIOMode : int32_t {
    kNONE   = 0,
    kINPUT  = 1,
    kOUTPUT = 2,
};

class Dims64 {
public:
    static constexpr int32_t MAX_DIMS = 8;
    int32_t nbDims = 0;
    int64_t d[MAX_DIMS] = {0};
};
using Dims = Dims64;

class ILogger {
public:
    enum class Severity : int32_t {
        kINTERNAL_ERROR = 0,
        kERROR          = 1,
        kWARNING        = 2,
        kINFO           = 3,
        kVERBOSE        = 4,
    };
    virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;
    virtual ~ILogger() = default;
};

class IExecutionContext {
public:
    virtual ~IExecutionContext() = default;
    virtual bool  setInputShape(AsciiChar const* name, Dims const& dims)  = 0;
    virtual Dims  getTensorShape(AsciiChar const* name) const             = 0;
    virtual bool  setTensorAddress(AsciiChar const* name, void* data)     = 0;
    virtual bool  enqueueV3(cudaStream_t stream)                          = 0;
};

class ICudaEngine {
public:
    virtual ~ICudaEngine() = default;
    virtual int32_t            getNbIOTensors() const                        = 0;
    virtual AsciiChar const*   getIOTensorName(int32_t index) const          = 0;
    virtual TensorIOMode       getTensorIOMode(AsciiChar const* name) const  = 0;
    virtual Dims               getTensorShape(AsciiChar const* name) const   = 0;
    virtual DataType           getTensorDataType(AsciiChar const* name) const= 0;
    virtual IExecutionContext* createExecutionContext()                      = 0;
};

class IRuntime {
public:
    virtual ~IRuntime() = default;
    virtual ICudaEngine* deserializeCudaEngine(void const* blob,
                                               std::size_t size) = 0;
};

// Defined in test/stubs/trt_stub.cpp so the check target links.
IRuntime* createInferRuntime(ILogger& logger) noexcept;

} // namespace nvinfer1
