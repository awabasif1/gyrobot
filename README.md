# reflex-cv

A two-wheeled robot that balances upright on its own and turns to track a
target through a camera - combining a custom CUDA/TensorRT vision pipeline
with a real-time embedded balance loop.

See [`BUILD_GUIDE.md`](./BUILD_GUIDE.md) for the full project explanation,
hardware list, and step-by-step build plan. See
[`docs/ENVIRONMENT.md`](./docs/ENVIRONMENT.md) for each team member's local
toolchain versions.

## Repo layout

```
reflex-cv/
├── CMakeLists.txt              # build config (targets both team GPUs)
├── BUILD_GUIDE.md              # full project explanation + plan
├── include/
│   ├── cuda_utils.cuh          # CUDA_CHECK error-handling macro
│   └── stage_profiler.cuh      # CUDA-events per-stage profiler
├── src/
│   ├── main.cpp                # OpenCV capture + pipeline orchestration
│   ├── preprocess.cu            # custom resize/color/normalize kernel
│   ├── preprocess.cuh
│   ├── trt_infer.cpp            # TensorRT engine wrapper
│   └── trt_infer.h
├── mcu/
│   ├── balance_controller.ino  # ESP32: IMU filter + PID + serial
│   └── SERIAL_PROTOCOL.md      # shared packet format
├── models/
│   ├── model.onnx              # committed - GPU-agnostic source model
│   └── build_engine.sh         # run locally per machine, output gitignored
└── docs/
    └── ENVIRONMENT.md
```

## Branches

- `main` - stable, working code only
- `vision-cuda` - CUDA/TensorRT pipeline track
- `balance-esp32` - embedded balance control track

Merge into `main` via pull request once a track is working, don't push to
`main` directly.

## Setup (each machine)

1. Install CUDA Toolkit, TensorRT, OpenCV, and CMake matching versions
   recorded (or to be recorded) in `docs/ENVIRONMENT.md`.
2. Run `nvidia-smi --query-gpu=compute_cap --format=csv` and make sure your
   GPU's compute capability is listed in `CMAKE_CUDA_ARCHITECTURES` in
   `CMakeLists.txt`.
3. Build the engine locally: `cd models && ./build_engine.sh model.onnx model.engine`
4. Build the project: `mkdir build && cd build && cmake .. && cmake --build .`
