# reflex-cv

A two-wheeled robot that balances upright on its own and turns to track a
target through a camera — combining a custom CUDA/TensorRT vision pipeline
with a real-time embedded balance loop.

See [`BUILD_GUIDE.md`](./BUILD_GUIDE.md) for the full project explanation,
hardware list, and step-by-step build plan. See
[`docs/ENVIRONMENT.md`](./docs/ENVIRONMENT.md) for each team member's local
toolchain versions, and [`mcu/SERIAL_PROTOCOL.md`](./mcu/SERIAL_PROTOCOL.md)
for the contract between the three boards.

## How the pieces fit

```
  ESP32-CAM                  Laptop (NVIDIA GPU)                Arduino Mega
  ─────────                  ───────────────────                ────────────
  camera ──MJPEG/WiFi──▶  capture → preprocess (CUDA)
                                  → inference (TensorRT)
                                  → decode + command shaping
          ◀──HTTP /cmd────────────┘
     │
     └──UART 115200──▶  balance loop: IMU → complementary filter
                        → PID @ 200Hz → motors
                        ──telemetry──▶ back up the same path
```

Three processors because the three jobs have incompatible timing needs. Vision
is variable-latency (20–40 ms per frame, plus whatever WiFi does today).
Balancing cannot tolerate variable latency at all. So the Mega runs its own
200 Hz loop and simply uses the most recent command it received — if a frame,
an inference pass, or a WiFi packet is late, the robot keeps balancing instead
of the whole system stalling. **The balance loop never waits on the vision
loop.**

## Repo layout

```
reflex-cv/
├── CMakeLists.txt              # builds the pipeline; tests build without a GPU
├── BUILD_GUIDE.md              # full project explanation + plan
├── include/
│   ├── cuda_utils.cuh          # CUDA_CHECK error-handling macro
│   ├── stage_profiler.cuh      # CUDA-events per-stage profiler (GPU stages)
│   └── host_profiler.h         # steady_clock profiler (CPU/network stages)
├── src/
│   ├── main.cpp                # capture thread + pipeline orchestration
│   ├── preprocess.cu/.cuh      # fused letterbox/BGR→RGB/normalise/HWC→CHW kernel
│   ├── preprocess_math.cuh     # geometry shared by kernel AND postprocessor
│   ├── trt_infer.cpp/.h        # TensorRT engine wrapper
│   ├── postprocess.cpp/.h      # YOLOv8 decode, NMS, target selection
│   ├── command_shaper.cpp/.h   # detection → smoothed lean/turn command
│   └── esp32_client.cpp/.h     # threaded HTTP command link (never blocks)
├── mcu/
│   ├── balance_controller.ino  # Mega: IMU + filter + PID + protocol
│   ├── balance_core.h          # that firmware's logic, Arduino-free + tested
│   ├── imu_calibrate.ino       # RUN FIRST: gyro bias + angle sign check
│   ├── uart_echo_test.ino      # link test; never touches the motors
│   └── SERIAL_PROTOCOL.md      # packet format shared by all three boards
├── esp32cam/
│   ├── CameraWebServer_uart_bridge.ino   # MJPEG + /cmd + /telemetry + UART
│   └── README.md               # flashing, board settings, what goes wrong
├── tools/
│   └── verify_preprocess.cpp   # GPU kernel vs OpenCV CPU reference
├── test/                       # host unit tests - no GPU, no TensorRT
│   ├── Makefile                # `make -C test`
│   └── stubs/                  # minimal NvInfer.h for type-checking only
├── models/
│   ├── export_onnx.py          # YOLOv8 → ONNX (commit the .onnx)
│   └── build_engine.sh         # ONNX → .engine, per machine, gitignored
└── docs/
    └── ENVIRONMENT.md
```

## Quickstart

### 1. Tests first — these need no GPU and no robot

```bash
make -C test
```

Four suites covering the geometry maths, the detector decoding, the command
shaping, and the whole balance-loop logic — including a closed-loop inverted
pendulum simulation that verifies the PID's sign convention. Run these before
touching hardware; they catch the class of bug that is miserable to diagnose
with a robot on the floor.

### 2. Model and engine

```bash
cd models
python3 export_onnx.py                      # once, produces model.onnx
./build_engine.sh model.onnx model.engine   # on EACH machine
```

A `.engine` is tied to one GPU architecture, TensorRT version and driver.
Never commit it, never copy it between machines.

### 3. Build the pipeline

```bash
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/path/to/TensorRT-10.x
cmake --build . -j
```

Check your GPU's compute capability is in `CMAKE_CUDA_ARCHITECTURES`:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

### 4. Verify the kernel against OpenCV before trusting it

```bash
./build/verify_preprocess --bench
```

### 5. Firmware

1. `mcu/imu_calibrate.ino` → **run this first**; it gives you the gyro bias and
   confirms the angle's sign. Nothing works until the sign is right.
2. `esp32cam/CameraWebServer_uart_bridge.ino` → set WiFi credentials, flash,
   confirm the stream in a browser.
3. `mcu/uart_echo_test.ino` → prove the UART link works, motors inert.
4. `mcu/balance_controller.ino` → the real thing. Tune P, then D, then I, with
   the wheels off the ground. Tuning notes are at the bottom of that file.

### 6. Run it

```bash
# Dry run: whole pipeline, robot switched off. Do this first.
./build/reflex_cv --stream http://192.168.1.50:81/stream --no-send

# For real
./build/reflex_cv --stream http://192.168.1.50:81/stream \
                  --engine models/model.engine --class 0
```

`--class 0` follows a person; `--class 32` follows a sports ball.
`--help` lists everything. Ctrl-C sends a zero command before exiting — without
that the Mega keeps acting on the last command it heard.

## Latency breakdown (fill in from a real run)

`reflex_cv` prints both timing tables on exit and a markdown version ready to
paste here. `--csv out.csv` dumps every sample.

| Stage | Mean (ms) | Min (ms) | p95 (ms) | Max (ms) | Notes |
|---|---|---|---|---|---|
| Capture wait (incl. WiFi transit + JPEG decode) | — | — | — | — | host-timed |
| Upload (H2D) | — | — | — | — | from pinned memory |
| Preprocess (CUDA) | — | — | — | — | before/after shared-memory tiling |
| TensorRT inference | — | — | — | — | before/after FP16 |
| Download (D2H) | — | — | — | — | |
| Postprocess + command shaping | — | — | — | — | host-timed |
| **End-to-end** | — | — | — | — | |

Measure preprocess twice — `verify_preprocess --bench` reports the
shared-memory kernel and the global-memory kernel side by side, which is the
honest before/after for the shared-memory optimisation.

## Branches

- `main` — stable, working code only
- `vision-cuda` — CUDA/TensorRT pipeline track
- `balance-esp32` — embedded balance control track

Merge into `main` via pull request once a track is working; don't push to
`main` directly.
