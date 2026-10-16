#include "trt_infer.h"
#include "cuda_utils.cuh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace reflex {

namespace {

size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: break;
    }
    // Newer TensorRT adds types (UINT8, FP8, INT64, BF16) whose enumerators do
    // not exist in every supported header version, so they are handled here by
    // value rather than by name. 4 bytes is the safe over-estimate for
    // allocation purposes; an engine actually using one of these for a top
    // level I/O tensor is rejected by the dtype checks in the constructor.
    return 4;
}

} // namespace

size_t dims_volume(const nvinfer1::Dims& d) {
    if (d.nbDims <= 0) return 0;
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) return 0;   // still dynamic / unresolved
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

std::string dims_to_string(const nvinfer1::Dims& d) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) os << "x";
        os << static_cast<long long>(d.d[i]);
    }
    os << "]";
    return os.str();
}

const char* dtype_name(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "fp32";
        case nvinfer1::DataType::kHALF:  return "fp16";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kINT8:  return "int8";
        case nvinfer1::DataType::kBOOL:  return "bool";
        default: return "other";
    }
}

void TrtEngine::Logger::log(Severity severity, const char* msg) noexcept {
    if (!verbose_ && severity > Severity::kWARNING) return;
    const char* tag = "[trt]";
    switch (severity) {
        case Severity::kINTERNAL_ERROR: tag = "[trt FATAL]"; break;
        case Severity::kERROR:          tag = "[trt ERROR]"; break;
        case Severity::kWARNING:        tag = "[trt warn]";  break;
        case Severity::kINFO:           tag = "[trt info]";  break;
        default:                        tag = "[trt dbg]";   break;
    }
    std::cerr << tag << " " << (msg ? msg : "") << "\n";
}

TrtEngine::TrtEngine(const std::string& engine_path, int batch,
                     bool verbose_log)
    : logger_(verbose_log), batch_(batch < 1 ? 1 : batch) {
    // --- read the serialised engine ---------------------------------------
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error(
            "cannot open engine file '" + engine_path +
            "'\n  Engines are not committed to the repo because they are "
            "GPU-specific.\n  Build one for this machine:  cd models && "
            "./build_engine.sh model.onnx model.engine");
    }
    const std::streamsize size = f.tellg();
    if (size <= 0) {
        throw std::runtime_error("engine file is empty: " + engine_path);
    }
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<size_t>(size));
    if (!f.read(blob.data(), size)) {
        throw std::runtime_error("failed reading engine file: " + engine_path);
    }

    // --- deserialise -------------------------------------------------------
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        throw std::runtime_error("nvinfer1::createInferRuntime returned null");
    }

    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) {
        free_all();
        throw std::runtime_error(
            "deserializeCudaEngine failed for '" + engine_path +
            "'\n  A serialised engine is tied to the GPU architecture, the "
            "TensorRT version\n  and the driver it was built against - it is "
            "not portable between the\n  two machines on this project. "
            "Rebuild it locally:\n      cd models && ./build_engine.sh "
            "model.onnx model.engine");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        free_all();
        throw std::runtime_error("createExecutionContext failed (out of GPU memory?)");
    }

    // --- pass 1: enumerate tensors, pin down the input shapes --------------
    const int n_io = engine_->getNbIOTensors();
    if (n_io <= 0) {
        free_all();
        throw std::runtime_error("engine reports no I/O tensors");
    }

    for (int i = 0; i < n_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        TrtTensor t;
        t.name     = name ? name : "";
        t.is_input = engine_->getTensorIOMode(name) ==
                     nvinfer1::TensorIOMode::kINPUT;
        t.dtype    = engine_->getTensorDataType(name);

        if (t.is_input) {
            nvinfer1::Dims d = engine_->getTensorShape(name);
            // A dynamic batch dimension is the one kind of dynamism worth
            // supporting: it is how batched frame processing is enabled
            // without rebuilding the engine.
            if (d.nbDims > 0 && d.d[0] < 0) d.d[0] = batch_;
            for (int k = 1; k < d.nbDims; ++k) {
                if (d.d[k] < 0) {
                    free_all();
                    throw std::runtime_error(
                        std::string("input '") + t.name +
                        "' has a dynamic non-batch dimension " +
                        dims_to_string(d) +
                        ".\n  Rebuild the engine with fixed spatial dims, e.g. "
                        "trtexec --shapes=" + t.name + ":" +
                        std::to_string(batch_) + "x3x640x640");
                }
            }
            if (!context_->setInputShape(name, d)) {
                free_all();
                throw std::runtime_error(
                    std::string("setInputShape rejected ") + dims_to_string(d) +
                    " for input '" + t.name +
                    "'.\n  Requested batch (" + std::to_string(batch_) +
                    ") is probably outside the engine's optimisation profile.");
            }
        }

        tensors_.push_back(t);
    }

    // --- pass 2: now that inputs are bound, output shapes resolve ----------
    for (auto& t : tensors_) {
        t.dims     = context_->getTensorShape(t.name.c_str());
        t.elements = dims_volume(t.dims);
        if (t.elements == 0) {
            free_all();
            throw std::runtime_error("could not resolve a concrete shape for tensor '" +
                                     t.name + "' " + dims_to_string(t.dims));
        }
        t.bytes = t.elements * dtype_size(t.dtype);

        CUDA_CHECK(cudaMalloc(&t.device, t.bytes));
        if (!context_->setTensorAddress(t.name.c_str(), t.device)) {
            free_all();
            throw std::runtime_error("setTensorAddress failed for tensor '" +
                                     t.name + "'");
        }
    }

    // --- identify the one input and the detection output -------------------
    int n_inputs = 0;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        if (tensors_[i].is_input) {
            ++n_inputs;
            input_idx_ = static_cast<int>(i);
        } else if (output_idx_ < 0 ||
                   tensors_[i].elements > tensors_[output_idx_].elements) {
            // Largest output wins. Some ONNX exports emit auxiliary outputs
            // alongside the detection tensor; the detections are always the
            // big one.
            output_idx_ = static_cast<int>(i);
        }
    }

    if (n_inputs != 1 || input_idx_ < 0 || output_idx_ < 0) {
        free_all();
        throw std::runtime_error(
            "expected exactly 1 input and at least 1 output, found " +
            std::to_string(n_inputs) + " inputs and " +
            std::to_string(static_cast<int>(tensors_.size()) - n_inputs) +
            " outputs.\n" + summary());
    }

    // --- sanity checks worth failing loudly on -----------------------------
    const TrtTensor& ti = in();
    const TrtTensor& to = out();

    if (ti.dtype != nvinfer1::DataType::kFLOAT) {
        free_all();
        throw std::runtime_error(
            std::string("engine input '") + ti.name + "' is " +
            dtype_name(ti.dtype) +
            ", but the preprocessing kernel writes fp32.\n  Rebuild the engine "
            "without --inputIOFormats=fp16:chw. Using --fp16 alone is fine and "
            "is what you want:\n  it runs the network in half precision while "
            "keeping the I/O tensors fp32.");
    }
    if (to.dtype != nvinfer1::DataType::kFLOAT) {
        free_all();
        throw std::runtime_error(
            std::string("engine output '") + to.name + "' is " +
            dtype_name(to.dtype) + ", but the postprocessor reads fp32.");
    }
    if (ti.dims.nbDims != 4) {
        free_all();
        throw std::runtime_error("expected a 4-D NCHW input, got " +
                                 dims_to_string(ti.dims));
    }
    if (ti.dims.d[1] != 3) {
        free_all();
        throw std::runtime_error(
            "expected 3 input channels, got " +
            std::to_string(static_cast<long long>(ti.dims.d[1])) + " " +
            dims_to_string(ti.dims));
    }
}

TrtEngine::~TrtEngine() { free_all(); }

void TrtEngine::free_all() noexcept {
    for (auto& t : tensors_) {
        if (t.device) {
            cudaFree(t.device);   // deliberately unchecked: this runs in a
            t.device = nullptr;   // destructor and on error paths
        }
    }
    // Reverse construction order: the context references the engine, which
    // references the runtime.
    delete context_; context_ = nullptr;
    delete engine_;  engine_  = nullptr;
    delete runtime_; runtime_ = nullptr;
}

void TrtEngine::enqueue(cudaStream_t stream) {
    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error(
            "IExecutionContext::enqueueV3 failed - inference was not queued");
    }
}

int TrtEngine::input_w() const {
    return static_cast<int>(in().dims.d[3]);
}
int TrtEngine::input_h() const {
    return static_cast<int>(in().dims.d[2]);
}

std::string TrtEngine::summary() const {
    std::ostringstream os;
    os << "TensorRT engine (" << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR
       << ") tensors:\n";
    for (const auto& t : tensors_) {
        os << "    " << (t.is_input ? "in  " : "out ") << t.name << "  "
           << dims_to_string(t.dims) << "  " << dtype_name(t.dtype) << "  "
           << (t.bytes / 1024) << " KiB\n";
    }
    return os.str();
}

} // namespace reflex
