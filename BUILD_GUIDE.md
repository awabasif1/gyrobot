# Vision-Guided Self-Balancing Robot — Build Guide

> **In one sentence:** a self-balancing two-wheel robot that sees a target
> through a camera and leans/turns to follow it, without ever falling over.

**Contents:** [Executive summary](#executive-summary) ·
[Skills this covers](#what-this-project-covers-mapped-to-what-roboticsai-companies-actually-screen-for) ·
[How it works](#how-it-works-the-see--decide--move-loop) ·
[Hardware list](#hardware-bill-of-materials) ·
[Step-by-step plan](#step-by-step-plan-from-absolute-zero-for-two-people) ·
[Technical checklist](#technical-requirements-checklist-visioncuda-half)

---

## Executive summary

**What you're building:** a two-wheeled robot that balances upright on its own
(like a tiny Segway), and can see a target through a camera and lean/turn
toward it. It's two systems in one: a real-time balance controller (embedded
electronics) and a GPU vision pipeline (computer vision + AI), connected by a
simple serial link.

**What "done" looks like:** you place the robot on the floor, it stands itself
up, and when you roll a ball or walk in front of it, it turns to track and
follow you — while never falling over, even while doing so.

**Why this project specifically:** most student CV projects stop at "detect
an object in a video." This one forces you to close the loop from *seeing*
something to a *physical machine acting on it in real time*, under a hard
constraint (don't fall over) that has nothing to do with AI at all. That
combination — perception, real-time control, and systems integration under a
physical constraint — is close to a scaled-down version of what a robotics
company's early-career engineers actually work on.

**Hardware note:** this build runs the vision/CUDA/TensorRT pipeline on an
onboard NVIDIA Jetson Nano (2GB) instead of a laptop, with an Arduino Mega
handling the balance loop, connected over a UART serial link rather than a
USB cable. That's the main upside of the Jetson version: the robot is fully
untethered and self-contained, running off its own battery. The tradeoff
runs the other way — the Nano 2GB has a tight RAM budget, so you'll need to
run headless (no desktop GUI), watch memory usage during TensorRT engine
builds, and possibly enable a swap file. See
[Hardware bill of materials](#hardware-bill-of-materials) for details.

## What this project covers, mapped to what robotics/AI companies actually screen for

| Topic you'll hit | What it looks like in this project | Why companies care |
|---|---|---|
| Embedded real-time control | Arduino Mega reading an IMU and running a PID loop at hundreds of Hz | Nearly every physical product (robots, drones, EVs, medical devices) has a real-time control loop underneath it |
| Sensor fusion | Combining accelerometer + gyro into one stable angle estimate | Every robot with more than one sensor needs this; it's a named skill on robotics job postings |
| Low-level GPU programming | Custom CUDA kernel for image preprocessing, explicit memory management | Signals you understand what's happening *under* PyTorch/TensorFlow, not just how to call them — this is what separates "used AI" from "built AI infrastructure" |
| Inference optimization | Converting a model to TensorRT, quantizing it, measuring the speedup | Directly NVIDIA's own product category; a resume line with real before/after latency numbers is rare and gets noticed |
| Computer vision | Detecting and tracking a moving target in real time | Baseline expectation for any perception role |
| Systems integration | Two independent processors (Mega + your GPU machine) talking over serial without one destabilizing the other | This is the actual day-to-day of robotics engineering — most of the hard problems are at the boundaries between subsystems, not inside any one of them |
| Performance measurement discipline | Per-stage latency breakdown with CUDA events, documented reasoning for kernel launch configs | Shows you think about *why* something is fast or slow, not just that it works |

## How it works: the see → decide → move loop

At runtime, this is the entire story, repeating continuously, many times per
second:

1. **See** — a camera on the Jetson Nano captures a frame, a custom CUDA
   kernel preprocesses it on the GPU, and a TensorRT-optimized model detects
   the target (a ball, a person, whatever you choose).
2. **Decide** — the detection's position is turned into a lean-angle offset
   and a turn-rate value, smoothed so it doesn't jump around frame to frame,
   and sent to the Arduino Mega over a UART serial link (Jetson GPIO
   TX/RX pins ↔ Mega TX/RX pins).
3. **Move** — the Mega takes that command and adds it to its own,
   much-faster internal balance loop (reading the IMU and running PID at
   200–500Hz), driving the motors to both stay upright *and* lean/turn
   toward the target.

The reason this is split across two separate processors rather than done all
in one place: camera/GPU work has variable timing — one frame might take
20ms, the next 40ms, depending on what's in view — while staying upright
cannot tolerate variable timing at all. If the "See" and "Decide" steps run a
little slow on a given frame, the Mega just keeps balancing on the last
command it received rather than the whole system stalling. The balance loop
never waits on the vision loop for anything.

**In short:** the Jetson is the eyes and brain deciding *where to go*; the
Arduino Mega is the reflexes making sure it doesn't fall down while getting
there.

**Untethered by design:** because the vision pipeline runs onboard the
Jetson rather than on a laptop, the whole robot is self-contained — the
Jetson talks to the Mega over a few GPIO wires (UART TX/RX + ground)
instead of a USB cable, and both boards run off the robot's own battery.
This is a step up from a laptop-tethered v1: no cable to trip over or keep
slack in during a demo, and it's what makes the "follow me across the room"
version of this project actually work. The one thing to plan for is power —
see the BOM below for how the Jetson and motors get fed from the same
battery without one starving the other.

## Hardware bill of materials

| Component | Purpose | Approx. cost |
|---|---|---|
| microSD card, 64GB+, A2/UHS-3 rated | JetPack OS + all your code/models — the Jetson boots from this, none is included with the board | ~$12–15 |
| 5V/3A USB-C power supply (or barrel-jack equivalent for older Nano revisions) | Dedicated power for the Jetson — underpowering it causes throttling/random shutdowns under GPU load, so don't share this off the motor battery | ~$10–15 |
| USB or CSI camera (e.g. OV9281 USB global-shutter variant, or a Raspberry Pi Camera Module v2 on the CSI port) | Frame capture — global shutter avoids rolling-shutter blur on fast motion | ~$15–40 |
| MPU6050 or MPU9250 IMU | Tilt angle + angular velocity sensing | ~$5 |
| TB6612FNG motor driver | Drives both motors, better efficiency than L298N | ~$5–8 |
| 2x DC gear motors (encoders a plus) | Drive wheels | ~$20–40 |
| Wheels + frame (3D printed, laser cut, or off-the-shelf chassis kit) | Structure — keep it **low and wide**, not tall and narrow, for easier balance; needs room to mount the Jetson plus its own battery | ~$20–40 |
| LiPo battery + regulator(s) | Power for motors + electronics — run the Jetson off its own regulated 5V/3A line, separate from the motor driver's supply, so motor current spikes don't brown out the Jetson | ~$20–30 |

**Already owned / not purchased:**
- Nvidia Jetson Nano 2GB — runs the CUDA/TensorRT vision pipeline onboard
- Arduino Mega — runs the balance loop
- Jumper wires (Jetson GPIO ↔ Mega, UART TX/RX + GND; no USB cable needed once flashed)

**Total: roughly $105–155**, mostly because the Jetson needs its own microSD
card and power supply that a laptop-based build wouldn't. If budget's still
tight, a regular (non-global-shutter) USB webcam works for ~$10–15 instead
of the CSI/global-shutter options — you'll get some motion blur on fast
pans, which mainly hurts tracking a fast-moving target, not the rest of the
pipeline.

## Step-by-step plan from absolute zero, for two people

This assumes **neither of you has done embedded electronics, CUDA, or
robotics before.** It's split into two tracks that run mostly in parallel, so
you're not blocked waiting on each other, with joint checkpoints where the
tracks meet. Realistic pace: this is a multi-week project, not a weekend one
— treat the "few weeks" hackathon version as Phases 0–2 only, with Phases 3–5
as stretch goals if time allows.

---

### Both of you, together — Week 0: setup and shared groundwork

1. Order all hardware from the BOM above — this has the longest lead time, so do it first, day one.
2. Flash the Jetson: download the JetPack SD card image for your Nano 2GB
   revision, write it to the microSD card (balenaEtcher or `dd` both work),
   boot the Nano, and run through NVIDIA's first-boot setup. JetPack ships
   CUDA, cuDNN, and TensorRT pre-installed, so there's no separate driver/CUDA
   Toolkit install to do — unlike a laptop-GPU setup. Also install: Arduino
   IDE (for the Mega, on whichever machine you use to program it), and on the
   Jetson itself, OpenCV (usually already present in JetPack) and a C++ build
   toolchain (CMake, g++, both included in JetPack's Linux image).
3. Set up the Jetson to run headless (SSH in over WiFi or Ethernet rather
   than keeping a monitor/keyboard attached) — with 2GB of RAM you want every
   spare megabyte going to your pipeline, not a desktop GUI. Also add a swap
   file (4–6GB on the microSD is fine) before you get to TensorRT engine
   builds in Person B's Step 4 — building an engine is memory-hungry and can
   fail or hang on 2GB without swap.
4. Both read through this whole document together so you have a shared mental model before splitting up. Draw the architecture diagram on a whiteboard in your own words — if you can't explain it to each other, re-read the architecture section.
5. Agree on the serial protocol now, even though nothing uses it yet: e.g. an ASCII line `"L:<lean_offset>,T:<turn_rate>\n"` from Jetson to Mega, and `"A:<angle>,F:<fall_flag>\n"` back. Write it down in a shared doc. Agreeing on this interface early is what lets you work independently without integration surprises later.

---

### Person A track — Balance / embedded (assuming zero embedded experience)

**Step 1 — Learn the absolute basics of the Arduino Mega.**
Follow a beginner Arduino tutorial: blink an LED, read a button, print
values to the serial monitor. Goal: comfortable uploading code and reading
serial output before touching any sensor.

**Step 2 — Get the IMU talking to the Mega.**
Wire the MPU6050 via I2C (4 wires: VCC, GND, SDA→pin 20, SCL→pin 21 on the
Mega). Use a basic library example to print raw accelerometer/gyro values to
serial. Goal: numbers change sensibly when you tilt the board by hand.

**Step 3 — Turn raw IMU data into a stable angle.**
Implement a complementary filter (a short, well-documented formula — search
"complementary filter IMU angle") combining accelerometer and gyro data.
Goal: a single angle number that's stable even when you shake the board a
little, not just noisy raw accelerometer output.

**Step 4 — Get motors spinning under code control.**
Wire the TB6612FNG to the Mega and to both motors. Write code that spins
each motor forward/backward at a given speed. Goal: both wheels respond
correctly and independently to commands, no balancing logic yet.

**Step 5 — Mount everything on the physical frame.**
Build/assemble the frame with motors, wheels, IMU mounted near the pivot
point, battery, and electronics all fixed in place. Goal: a physical object
that can be held upright and has all electronics working while assembled.

**Step 6 — Write and tune the PID balance loop.**
Combine steps 3 and 4: error = 0° − current angle, PID output drives motor
speed/direction. Start with only the P term, increase until it oscillates,
back off ~30%, then add D to damp the oscillation, then a small I term.
Goal: the robot recovers from a gentle push and holds itself upright,
unassisted, for at least 10+ seconds.

**Step 7 — Add the serial command interface.**
Implement the protocol agreed on in Week 0: parse incoming lean/turn
commands and add them to the PID setpoint each loop; send telemetry back
over a dedicated hardware serial port (e.g. `Serial1` on the Mega, wired to
the Jetson's UART GPIO pins) — keep `Serial` (the USB port) free for
debugging over your laptop while you're bench-testing, separate from the
Jetson link. Test with a throwaway script (even just typing values into a
serial terminal) — don't wait for Person B's pipeline to test this.

---

### Person B track — Vision / CUDA (assuming zero CUDA/GPU programming experience)

**Step 1 — Learn C++ and OpenCV basics for video.**
If C++ is new, spend real time here — a shaky foundation here compounds
later. Goal: a small C++ program running **on the Jetson** that opens your
camera (USB via `cv::VideoCapture(0)`, or CSI via a GStreamer pipeline
string if you're using a Pi camera module) with OpenCV and displays it —
either over X11 forwarding via SSH, or by saving a frame to disk and
`scp`-ing it back to check, since you're running headless.

**Step 2 — Learn CUDA fundamentals.**
Work through an introductory CUDA C++ tutorial (NVIDIA's own "An Easy
Introduction to CUDA" is a good start). Write and run the classic toy
examples: vector addition, then a simple image operation like grayscale
conversion. Goal: understand what a kernel, a thread, a block, and a grid
are, and how `cudaMalloc`/`cudaMemcpy` move data — this project explicitly
requires explicit memory management, no shortcuts.

**Step 3 — Write the custom preprocessing kernel.**
Build a CUDA kernel that takes a captured frame and does resize + color
conversion + normalization, entirely on GPU (`src/preprocess.cu` in this
repo is the target — ask for it once you're at this stage). Add shared
memory tile caching for the resize step. Goal: verify correctness by
comparing against OpenCV's CPU equivalent on a test image before trusting
your kernel's output.

**Step 4 — Get a detection model exported and running via TensorRT.**
Take a small pretrained detector (YOLOv8n is a reasonable choice — on a
Nano 2GB, avoid anything larger; even YOLOv8n may need FP16 to run
comfortably), export it to ONNX, then convert to a TensorRT engine using
`trtexec` (included in JetPack). This conversion step is the most
memory-hungry part of the whole project on a 2GB board — make sure your
swap file from Week 0 is active, and expect it to take noticeably longer
than it would on a desktop GPU. Write a C++ wrapper that loads the engine
and runs inference on a single test image. Goal: correct detections on a
static image before worrying about real-time video.

**Step 5 — Wire capture → preprocess → inference into one pipeline.**
Combine steps 1, 3, and 4 into a continuous loop processing live camera
frames. Add CUDA-events-based timing around each stage (capture, preprocess,
inference) so you have real numbers, not guesses. Goal: a live window
showing detections with a per-stage latency readout.

**Step 6 — Extract a target and shape it into a command.**
From the detection output, compute the target's position, convert it into a
lean/turn command, and apply a low-pass filter/rate limiter so it doesn't
jump around frame to frame. Goal: printed command values that move smoothly
and sensibly as you move a test object in front of the camera.

**Step 7 — Add the serial command interface.**
Implement the same protocol from Week 0, sending commands to the Mega over
the Jetson's UART GPIO pins (`/dev/ttyTHS1` on most Nano carrier boards —
confirm the pin numbers for your specific board) and reading telemetry
back. Note the Jetson's GPIO UART runs at 3.3V logic — check your Mega's
RX/TX voltage and add a level shifter if needed, since the Mega's I/O is
5V and feeding 5V into the Jetson's UART pin can damage it. Test against a
throwaway Mega script that just echoes what it receives — don't wait for
Person A's PID loop to be finished to test this.

---

### Both of you, together — integration

1. **Joint checkpoint 1:** Person A has a robot that balances standalone
   (their Step 6 done). Person B has a pipeline that prints correct commands
   from live video (their Step 6 done). Neither has touched the other's code
   yet — verify both independently before combining.
2. **Joint checkpoint 2:** connect the Jetson and Mega over the UART link
   (both Step 7s done) — first with the robot *not* trying to balance yet,
   just confirming commands and telemetry flow correctly both directions,
   and double-check the level-shifting from Person B's Step 7 before
   plugging anything in.
3. **Full integration:** run everything together. Expect this to take real
   tuning time — the low-pass filter from Person B's Step 6 will likely need
   adjusting against the real PID loop, not just against printed numbers.
   Keep the balance PID itself untouched; only adjust how incoming vision
   commands are shaped.
4. **Fallback decision point:** decide together, before you're out of time,
   what you'll demo if full tracking isn't reliable — a robot that balances
   perfectly and does one simple, robust behavior (e.g. leans toward
   whichever side a bright/colored object appears on) is a stronger demo
   than a fragile "impressive" one.

---

## Technical requirements checklist (vision/CUDA half)

| Requirement | Where it lives |
|---|---|
| Custom CUDA kernels in C++, no library abstractions hiding GPU work | `src/preprocess.cu` |
| Explicit `cudaMalloc`/`cudaMemcpy`, no managed memory | throughout `src/` |
| Custom preprocessing kernel (color conversion / normalization) on GPU | `src/preprocess.cu` |
| TensorRT for inference, not raw PyTorch/TensorFlow | `src/trt_infer.cpp` / `.h` |
| Batched frame processing | batch dimension in `src/preprocess.cu` launch config |
| CUDA events for per-stage latency profiling | `include/stage_profiler.cuh` |
| Shared memory usage in at least one kernel | `src/preprocess.cu` (tile caching for bilinear resize) |
| Grid/block dimension tuning with documented reasoning | comments in `src/preprocess.cu` |
| OpenCV in C++ for frame capture | `src/main.cpp` |
| End-to-end latency breakdown per stage, documented in README | Section below |
| Error handling on all CUDA calls | `include/cuda_utils.cuh`, `CUDA_CHECK` macro used everywhere |
| CMake/Makefile build system | `CMakeLists.txt` |

## Latency breakdown (fill in once running)

| Stage | Mean (ms) | Min (ms) | Max (ms) | Notes |
|---|---|---|---|---|
| Capture | — | — | — | |
| Preprocess (CUDA) | — | — | — | before/after shared-mem optimization |
| TensorRT inference | — | — | — | before/after FP16/INT8 quantization |
| Post-process + command shaping | — | — | — | |
| **End-to-end** | — | — | — | |

## Project file structure

```
reflex-cv/
├── CMakeLists.txt
├── README.md                    (this file)
├── include/
│   ├── cuda_utils.cuh           # CUDA_CHECK error-handling macro
│   └── stage_profiler.cuh       # CUDA-events per-stage profiler
├── src/
│   ├── main.cpp                 # OpenCV capture + pipeline orchestration
│   ├── preprocess.cu            # custom resize/color/normalize kernel
│   ├── preprocess.cuh
│   ├── trt_infer.cpp            # TensorRT engine wrapper
│   └── trt_infer.h
└── mcu/
    ├── balance_controller.ino   # Arduino Mega: IMU filter + PID + serial parsing
    └── SERIAL_PROTOCOL.md       # packet format shared by both sides
```

*(Files not yet created will be added as the project progresses — `main.cpp`,
`preprocess.cu`, `trt_infer.*`, and the `mcu/` directory are next.)*

## Talking points for interviews / recruiters

- "We wrote a custom CUDA kernel for resize+normalize instead of using
  OpenCV's GPU module, using shared memory tile caching, which got
  preprocessing from X ms to Y ms."
- "We measured every pipeline stage independently with CUDA events rather
  than wall-clock timing, so we knew exactly where latency was going."
- "Converting to TensorRT with FP16 quantization cut inference time from
  X ms to Y ms, which mattered because it's the difference between the
  robot reacting in time and not."
- "We ran the full CUDA/TensorRT pipeline onboard a Jetson Nano with 2GB of
  RAM, not on a desktop GPU — which meant managing memory tightly enough to
  build and run a TensorRT engine within that budget, and running the whole
  system headless."
- "We deliberately separated the safety-critical balance loop from the
  vision pipeline so that GPU latency variance couldn't destabilize the
  robot."
