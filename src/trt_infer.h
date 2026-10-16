#pragma once
// ---------------------------------------------------------------------------
// Thin owning wrapper around one deserialised TensorRT engine.
//
// Scope on purpose: load an engine, allocate its device buffers once, and
// enqueue it on a stream. No per-frame allocation, no hidden synchronisation,
// no copies the caller did not ask for. The preprocessing kernel writes
// straight into this object's input buffer, so between preprocess and
// inference nothing moves through host memory at all.
//
// Memory is explicitly cudaMalloc'd rather than using managed/unified memory.
// That is a project requirement, but it is also the right call here: unified
// memory would let page migration happen at unpredictable moments inside the
// inference call, which is precisely the latency variance this pipeline exists
// to measure.
// ---------------------------------------------------------------------------

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

// The name-based tensor API (setTensorAddress/enqueueV3) replaced the old
// index-based binding API in TensorRT 8.5 and is the only one present in
// TensorRT 10. Supporting both doubles the surface area of this file for the
// sake of a release line that predates the project, so the requirement is
// stated up front instead.
#if defined(NV_TENSORRT_MAJOR) &&                                              \
    (NV_TENSORRT_MAJOR < 8 ||                                                  \
     (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5))
#error "reflex-cv needs TensorRT 8.5 or newer (it uses the name-based tensor API: setTensorAddress + enqueueV3). Install a newer standalone TensorRT SDK and point TENSORRT_ROOT at it."
#endif

namespace reflex {

// One engine input or output, with its device allocation.
struct TrtTensor {
    std::string        name;
    nvinfer1::Dims     dims{};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    size_t             elements = 0;
    size_t             bytes    = 0;
    bool               is_input = false;
    void*              device   = nullptr;
};

class TrtEngine {
public:
    // Loads and deserialises `engine_path`, resolves shapes for `batch`, and
    // allocates every input and output on the device.
    //
    // Throws std::runtime_error with an actionable message on any failure -
    // most usefully when the engine was built for a different GPU or a
    // different TensorRT version, which is the single most common way this
    // stage breaks when two people share a repo.
    TrtEngine(const std::string& engine_path, int batch = 1,
              bool verbose_log = false);
    ~TrtEngine();

    TrtEngine(const TrtEngine&)            = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Queues inference on `stream`. Returns without waiting; the caller
    // synchronises (or enqueues the output copy on the same stream) when it
    // actually needs the result.
    void enqueue(cudaStream_t stream);

    // --- the buffers the rest of the pipeline writes to and reads from ---
    float*       input_device()  { return static_cast<float*>(in().device); }
    const float* output_device() const {
        return static_cast<const float*>(out().device);
    }

    const TrtTensor& in()  const { return tensors_[input_idx_]; }
    const TrtTensor& out() const { return tensors_[output_idx_]; }

    // Model input geometry, read from the engine rather than hard-coded, so
    // swapping in a 416x416 or 1280x1280 model needs no code change.
    int input_w()  const;
    int input_h()  const;
    int batch()    const { return batch_; }

    size_t output_elements() const { return out().elements; }
    size_t output_bytes()    const { return out().bytes; }

    const std::vector<TrtTensor>& tensors() const { return tensors_; }

    // Human-readable dump of every tensor's name/shape/dtype. Printed once at
    // startup; it is the fastest way to catch "this engine is not the model I
    // thought it was".
    std::string summary() const;

private:
    // TensorRT logs through this. Severity is filtered here rather than at the
    // call sites so a noisy engine build does not drown the latency output.
    class Logger : public nvinfer1::ILogger {
    public:
        explicit Logger(bool verbose) : verbose_(verbose) {}
        void log(Severity severity, const char* msg) noexcept override;
    private:
        bool verbose_;
    };

    Logger                      logger_;
    nvinfer1::IRuntime*         runtime_ = nullptr;
    nvinfer1::ICudaEngine*      engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    std::vector<TrtTensor> tensors_;
    int  input_idx_  = -1;
    int  output_idx_ = -1;
    int  batch_      = 1;

    void free_all() noexcept;
};

// Utility shared with main.cpp's startup logging and with the output-layout
// inference: product of all dimensions, or 0 if any is negative (unresolved).
size_t dims_volume(const nvinfer1::Dims& d);
std::string dims_to_string(const nvinfer1::Dims& d);
const char* dtype_name(nvinfer1::DataType t);

} // namespace reflex
