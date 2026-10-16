// Minimal definition so the stub compile-check can also LINK, which catches a
// different class of mistake than compiling alone (missing symbols, wrong
// overload selected). See test/stubs/NvInfer.h for what this is and is not.
//
// It returns nullptr, so a binary linked against it is not runnable - and
// trt_infer.cpp's own null check turns that into its normal "createInferRuntime
// returned null" error rather than a crash.

#include "NvInfer.h"

namespace nvinfer1 {

IRuntime* createInferRuntime(ILogger& logger) noexcept {
    (void)logger;
    return nullptr;
}

} // namespace nvinfer1
