A two-wheeled self-balancing robot that also turns to follow a target seen by a camera. The vision side runs on a laptop GPU (custom CUDA preprocessing plus a TensorRT model). The balance loop runs on a microcontroller. The two talk over serial.

The full explanation, parts list, and build plan are in [`BUILD_GUIDE.md`](./BUILD_GUIDE.md). Each person's toolchain versions are listed in [`docs/ENVIRONMENT.md`](./docs/ENVIRONMENT.md).

## Layout

```
reflex-cv/
├── CMakeLists.txt              build config, targets both team GPUs
├── BUILD_GUIDE.md
├── include/
│   ├── cuda_utils.cuh          CUDA_CHECK error macro
│   └── stage_profiler.cuh      per-stage timing with CUDA events
├── src/
│   ├── main.cpp                camera capture (OpenCV), runs the pipeline
│   ├── preprocess.cu           resize / color convert / normalize kernel
│   ├── preprocess.cuh
│   ├── trt_infer.cpp           TensorRT engine wrapper
│   └── trt_infer.h
├── mcu/
│   ├── balance_controller.ino  IMU filter, PID, serial
│   └── SERIAL_PROTOCOL.md      packet format both sides use
├── models/
│   ├── model.onnx              source model, works on any GPU
│   └── build_engine.sh         builds the engine locally (output is gitignored)
└── docs/
    └── ENVIRONMENT.md
```

## Branches

- `main`: working code only
- `vision-cuda`: CUDA/TensorRT work
- `balance-esp32`: balance control work

Don't push to `main`. Open a pull request once a branch works.

## Building

Do this on each machine.

1. Install the CUDA Toolkit, TensorRT, OpenCV, and CMake. Versions should match `docs/ENVIRONMENT.md`; add yours there if it's empty.
2. Check your GPU's compute capability:
   `nvidia-smi --query-gpu=compute_cap --format=csv`
   Make sure that value is in `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.
3. Build the TensorRT engine. Engines are tied to the GPU they're built on, so you have to do this on every machine:
   `cd models && ./build_engine.sh model.onnx model.engine`
4. Build the project:
   `mkdir build && cd build && cmake .. && cmake --build .`
