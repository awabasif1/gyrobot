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
serial link.

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

**Hardware note (revised):** the original plan for this build ran the
vision/CUDA/TensorRT pipeline onboard an NVIDIA Jetson Nano 2GB. As of
ordering, Jetson hardware (Nano and Orin alike) is severely supply-constrained
and running 2–4x MSRP everywhere it was checked — so this build instead runs
the CUDA/TensorRT pipeline on a laptop with a discrete NVIDIA GPU, with the
robot itself carrying only a small ESP32-CAM board for image capture and a
WiFi link back to the laptop. The Arduino Mega still handles the balance loop
locally exactly as before, connected to the ESP32-CAM over UART rather than
directly to a Jetson's GPIO pins. See [How it works](#how-it-works-the-see--decide--move-loop)
for the architecture and [Hardware bill of materials](#hardware-bill-of-materials)
for what actually got ordered.

## What this project covers, mapped to what robotics/AI companies actually screen for

| Topic you'll hit | What it looks like in this project | Why companies care |
|---|---|---|
| Embedded real-time control | Arduino Mega reading an IMU and running a PID loop at hundreds of Hz | Nearly every physical product (robots, drones, EVs, medical devices) has a real-time control loop underneath it |
| Sensor fusion | Combining accelerometer + gyro into one stable angle estimate | Every robot with more than one sensor needs this; it's a named skill on robotics job postings |
| Low-level GPU programming | Custom CUDA kernel for image preprocessing, explicit memory management | Signals you understand what's happening *under* PyTorch/TensorFlow, not just how to call them — this is what separates "used AI" from "built AI infrastructure" |
| Inference optimization | Converting a model to TensorRT, quantizing it, measuring the speedup | Directly NVIDIA's own product category; a resume line with real before/after latency numbers is rare and gets noticed |
| Computer vision | Detecting and tracking a moving target in real time | Baseline expectation for any perception role |
| Systems integration | Three independent boards (laptop, ESP32-CAM, Mega) talking across a WiFi hop and a UART hop without one destabilizing another | This is the actual day-to-day of robotics engineering — most of the hard problems are at the boundaries between subsystems, not inside any one of them |
| Performance measurement discipline | Per-stage latency breakdown with CUDA events, documented reasoning for kernel launch configs | Shows you think about *why* something is fast or slow, not just that it works |

## How it works: the see → decide → move loop

At runtime, this is the entire story, repeating continuously, many times per
second:

1. **See** — the ESP32-CAM on the robot captures a frame and streams it over
   WiFi to the laptop. There, a custom CUDA kernel preprocesses it on the
   laptop's GPU, and a TensorRT-optimized model detects the target (a ball, a
   person, whatever you choose).
2. **Decide** — the detection's position is turned into a lean-angle offset
   and a turn-rate value, smoothed so it doesn't jump around frame to frame,
   and sent back over WiFi to the ESP32-CAM, which relays it verbatim over a
   UART serial link to the Arduino Mega (ESP32-CAM UART pins ↔ Mega TX/RX
   pins).
3. **Move** — the Mega takes that command and adds it to its own,
   much-faster internal balance loop (reading the IMU and running PID at
   200–500Hz), driving the motors to both stay upright *and* lean/turn
   toward the target.

The reason perception and balance are split across separate processors rather
than done in one place: camera/GPU work has variable timing — one frame might
take 20ms, the next 40ms, and now there's WiFi round-trip on top of that —
while staying upright cannot tolerate variable timing at all. If a frame,
inference pass, or WiFi packet runs slow or drops entirely, the Mega just
keeps balancing on the last command it received rather than the whole system
stalling. The balance loop never waits on the vision loop, or the network,
for anything.

**In short:** the laptop is the eyes and brain deciding *where to go*; the
ESP32-CAM is just a camera with a WiFi antenna; the Arduino Mega is the
reflexes making sure it doesn't fall down while getting there.

**WiFi-tethered by design (this build):** because the vision pipeline runs on
the laptop rather than onboard the robot, the robot needs the laptop powered
on and in WiFi range any time vision-dependent behavior matters. This is the
real tradeoff versus the original Jetson-onboard plan — the robot is no
longer fully self-contained. What it buys back: dramatically lower cost and
no dependence on Jetson stock/pricing, a laptop GPU with far more headroom
than a 2GB Jetson Nano ever had (no headless/swap-file juggling to build a
TensorRT engine), and a normal desktop development environment instead of
cross-compiling and debugging over SSH to an embedded board. The balance loop
itself is completely unaffected by any of this — see the timing argument
above.

## Hardware bill of materials

All prices below are **live Amazon.ca listings in CAD**, verified 2026-09-15
(excludes shipping/tax). Exact product links are in the companion
`Robot_Hardware_BOM.xlsx` spreadsheet — treat that as the source of truth for
ordering; this table is the quick-reference version.

| Component | Purpose | Approx. cost (CAD) |
|---|---|---|
| ESP32-CAM MB Development Board (OV2640 camera + WiFi/BT, integrated USB programmer) | Onboard camera + WiFi bridge — replaces the Jetson entirely. Streams frames to the laptop, relays commands back to the Mega over UART | ~$18 |
| MPU6050 IMU | Tilt angle + angular velocity sensing for the balance loop | ~$21 |
| TB6612FNG motor driver | Drives both motors, better efficiency than an L298N | ~$12 |
| 2x DC gear motors w/ encoder + 65mm wheel | Drive wheels | ~$51 |
| Wheels + frame — 3D printed *or* off-the-shelf chassis kit | Structure — keep it **low and wide**, not tall and narrow, for easier balance; needs room to mount the boards and the battery. See the companion SolidWorks/3D-print guide if you're printing your own | ~$20–45 |
| LiPo battery (2S, 5200mAh) | Power for motors + electronics | ~$29 |
| UBEC 5V/3A regulator | Regulated 5V rail powering the Mega, the ESP32-CAM, and the motor driver's logic side — kept separate from the motors' raw LiPo line so motor current spikes don't brown out the electronics | ~$12 |
| LiPo balance charger | **Required, not optional** — a non-balancing charger risks overcharging one cell, a real fire hazard with LiPo chemistry | ~$52 |
| Connector & heat-shrink kit | Bridges the battery's connector type to the charger's, plus wiring the UBEC/motor-driver leads cleanly | ~$19 |
| M2/M3 standoff & screw kit | Mounts the Mega, ESP32-CAM, IMU, and motor driver to the chassis | ~$19 |

**Already owned / not purchased:**
- Arduino Mega — runs the balance loop
- Jumper wires (ESP32-CAM ↔ Mega, UART TX/RX + GND)
- A laptop with a discrete NVIDIA GPU — runs the CUDA/TensorRT vision pipeline; this is now a required piece of the system, just not a "robot part"
- Basic tools (soldering iron, multimeter, wire strippers) — assumed available at a school/library makerspace

**Total: ~$277 CAD** for everything actually purchased. This is a real jump
from the original ~$105–155 USD *estimate* in the first draft of this
document — that number assumed Jetson-era part prices that turned out not to
reflect 2026 reality (Jetson boards alone were quoted north of $800 CAD when
checked live), and it also didn't include the LiPo charger, connector kit, or
mounting hardware, all of which are genuinely required, not optional extras.

## Step-by-step plan from absolute zero, for two people

This assumes **neither of you has done embedded electronics, CUDA, or
robotics before.** It's split into two tracks that run mostly in parallel, so
you're not blocked waiting on each other, with joint checkpoints where the
tracks meet. Realistic pace: this is a multi-week project, not a weekend one
— treat the "few weeks" hackathon version as Phases 0–2 only, with Phases 3–5
as stretch goals if time allows.

---

### Both of you, together — Week 0: setup and shared groundwork

1. Parts are ordered (see the BOM above / the spreadsheet) — this had the
   longest lead time, so it's already underway.
2. Flash the ESP32-CAM: in the Arduino IDE, install the ESP32 board support
   package, then open the built-in **CameraWebServer** example sketch
   (File → Examples → ESP32 → Camera → CameraWebServer). Set your WiFi
   credentials in the sketch, select the correct camera model
   (`AI_THINKER` for most ESP32-CAM MB boards), and flash it. Out of the box
   this gives you a live MJPEG stream at `http://<esp32-ip>/stream` — no
   custom streaming code required. You'll extend this sketch later (Person
   B's Step 7) to also relay commands to the Mega over UART.
3. Set up your laptop's CUDA/cuDNN/TensorRT environment: install the NVIDIA
   driver, then CUDA Toolkit and cuDNN versions that match it, then the
   standalone TensorRT SDK (not the JetPack-bundled one — that only exists
   on Jetson hardware). This is more setup steps than JetPack's all-in-one
   install, but each piece is a normal desktop install with no ARM
   cross-compilation and no 2GB RAM ceiling to work around.
4. Install the Arduino IDE (for the Mega and the ESP32-CAM — same IDE, two
   different board targets) and OpenCV on the laptop (for Person B's track).
5. Both read through this whole document together so you have a shared
   mental model before splitting up. Draw the architecture diagram on a
   whiteboard in your own words — if you can't explain it to each other,
   re-read the architecture section.
6. Agree on the protocol now, even though nothing uses it yet. There are two
   hops: **laptop → ESP32-CAM** (WiFi) and **ESP32-CAM → Mega** (UART). Keep
   the Mega-facing format identical to what it would have been with a direct
   Jetson link, so Person A's whole track is unaffected by this
   architecture change:
   - Laptop → ESP32-CAM: an HTTP GET to a custom endpoint you'll add to the
     CameraWebServer sketch, e.g. `http://<esp32-ip>/cmd?lean=<x>&turn=<y>`.
     HTTP GET is simple to test by hand (curl or a browser) before any
     robot code exists.
   - ESP32-CAM → Mega: the same ASCII line format either way —
     `"L:<lean_offset>,T:<turn_rate>\n"` — which the ESP32 firmware forwards
     verbatim onto its UART pins after parsing the HTTP request.
   - Mega → ESP32-CAM → laptop (telemetry): `"A:<angle>,F:<fall_flag>\n"`
     over the same UART, which the ESP32 firmware can expose back to the
     laptop however's convenient (e.g. a `/telemetry` endpoint it caches
     and serves on request).
   Write this down in a shared doc. Agreeing on it early is what lets you
   work independently without integration surprises later.

---

### Person A track — Balance / embedded (assuming zero embedded experience)

*(Unaffected by the Jetson → laptop/ESP32-CAM change — your whole track reads
and writes the same UART lines regardless of what's on the other end of the
wire.)*

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
the **ESP32-CAM's** spare UART pins) — keep `Serial` (the USB port) free for
debugging over your laptop while you're bench-testing, separate from the
ESP32-CAM link. Note the ESP32-CAM's GPIO runs at 3.3V logic while the
Mega's I/O is 5V — add a level shifter between them, since feeding 5V into
the ESP32's UART pin can damage it. Test with a throwaway script (even just
typing values into a serial terminal) — don't wait for Person B's pipeline
to test this.

---

### Person B track — Vision / CUDA (assuming zero CUDA/GPU programming experience)

*(This is the track most affected by the platform switch — you're now
targeting the laptop's GPU and pulling frames over the network instead of
from a local camera. The CUDA/TensorRT engineering itself is identical
either way; only the capture source and command destination change.)*

**Step 1 — Learn C++ and OpenCV basics for video.**
If C++ is new, spend real time here — a shaky foundation here compounds
later. Goal: a small C++ program running **on the laptop** that opens the
ESP32-CAM's MJPEG stream with OpenCV
(`cv::VideoCapture("http://<esp32-ip>/stream")`) and displays it in a
window. Since this all runs locally on the laptop now (no SSH, no headless
Jetson), you get a normal debugger and a normal display — a real quality-of-
life upgrade over the original plan.

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
Take a small pretrained detector (YOLOv8n is a reasonable choice) export it
to ONNX, then convert to a TensorRT engine using `trtexec` (part of the
standalone TensorRT SDK you installed in Week 0). Your laptop's discrete GPU
almost certainly has more headroom than the Jetson Nano 2GB this project
originally targeted, so you have more room to experiment with model size and
precision (FP16 is still worth doing for the latency win, but it's no longer
the only way to make the model fit). Write a C++ wrapper that loads the
engine and runs inference on a single test image. Goal: correct detections
on a static image before worrying about real-time video.

**Step 5 — Wire capture → preprocess → inference into one pipeline.**
Combine steps 1, 3, and 4 into a continuous loop processing live frames from
the ESP32-CAM's network stream. Add CUDA-events-based timing around each
stage (capture, preprocess, inference) so you have real numbers, not
guesses. Goal: a live window showing detections with a per-stage latency
readout.

**Step 6 — Extract a target and shape it into a command.**
From the detection output, compute the target's position, convert it into a
lean/turn command, and apply a low-pass filter/rate limiter so it doesn't
jump around frame to frame. Goal: printed command values that move smoothly
and sensibly as you move a test object in front of the camera.

**Step 7 — Send commands to the ESP32-CAM, and extend its firmware to relay them.**
Two pieces here:
- **Laptop side:** implement the HTTP GET call from Week 0's protocol,
  fired once per shaped command from Step 6.
- **ESP32-CAM side:** extend the CameraWebServer sketch with a small custom
  HTTP handler for `/cmd` that parses the `lean`/`turn` query parameters,
  formats them into the agreed ASCII line, and writes it to `Serial1` (or
  whichever hardware UART pins you wired to the Mega). Add the matching
  `/telemetry` handler that returns whatever the Mega last sent back.
Test the ESP32 side against a throwaway Mega script that just echoes what it
receives over UART — don't wait for Person A's PID loop to be finished to
test this.

---

### Both of you, together — integration

1. **Joint checkpoint 1:** Person A has a robot that balances standalone
   (their Step 6 done). Person B has a pipeline that prints correct commands
   from live video (their Step 6 done). Neither has touched the other's code
   yet — verify both independently before combining.
2. **Joint checkpoint 2a — WiFi link:** confirm the laptop can reliably pull
   the ESP32-CAM's video stream and round-trip a `/cmd` request, on its own,
   with nothing plugged into the Mega yet.
3. **Joint checkpoint 2b — UART link:** connect the ESP32-CAM and Mega over
   UART (Person A's and Person B's Step 7s both done) — first with the
   robot *not* trying to balance yet, just confirming commands and
   telemetry flow correctly both directions, and double-check the
   level-shifting from Person A's Step 7 before plugging anything in.
4. **Full integration:** run everything together. Expect this to take real
   tuning time — the low-pass filter from Person B's Step 6 will likely need
   adjusting against the real PID loop and real WiFi latency, not just
   against printed numbers. Keep the balance PID itself untouched; only
   adjust how incoming vision commands are shaped.
5. **Fallback decision point:** decide together, before you're out of time,
   what you'll demo if full tracking isn't reliable — a robot that balances
   perfectly and does one simple, robust behavior (e.g. leans toward
   whichever side a bright/colored object appears on) is a stronger demo
   than a fragile "impressive" one. Also decide your fallback if the WiFi
   link itself is flaky in the demo room (a phone hotspot with known-good
   signal, tested in advance, is cheap insurance).

---

## Technical requirements checklist (vision/CUDA half)

*(Unchanged by the platform switch — all of this still applies, it just runs
on the laptop's GPU instead of a Jetson's.)*

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
| OpenCV in C++ for frame capture (now from a network stream, not a local device) | `src/main.cpp` |
| End-to-end latency breakdown per stage, documented in README | Section below |
| Error handling on all CUDA calls | `include/cuda_utils.cuh`, `CUDA_CHECK` macro used everywhere |
| CMake/Makefile build system | `CMakeLists.txt` |

## Latency breakdown (fill in once running)

| Stage | Mean (ms) | Min (ms) | Max (ms) | Notes |
|---|---|---|---|---|
| Network (ESP32-CAM → laptop, WiFi frame transit) | — | — | — | new stage vs. the original onboard-Jetson design |
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
│   ├── main.cpp                 # OpenCV network-stream capture + pipeline orchestration
│   ├── preprocess.cu            # custom resize/color/normalize kernel
│   ├── preprocess.cuh
│   ├── trt_infer.cpp            # TensorRT engine wrapper
│   └── trt_infer.h
├── esp32cam/
│   └── CameraWebServer_uart_bridge.ino   # stock CameraWebServer + /cmd + /telemetry + UART relay
└── mcu/
    ├── balance_controller.ino   # Arduino Mega: IMU filter + PID + serial parsing
    └── SERIAL_PROTOCOL.md       # packet format shared by all three boards
```

*(Files not yet created will be added as the project progresses — `main.cpp`,
`preprocess.cu`, `trt_infer.*`, `esp32cam/`, and the `mcu/` directory are
next.)*

## Talking points for interviews / recruiters

- "We wrote a custom CUDA kernel for resize+normalize instead of using
  OpenCV's GPU module, using shared memory tile caching, which got
  preprocessing from X ms to Y ms."
- "We measured every pipeline stage independently with CUDA events rather
  than wall-clock timing, so we knew exactly where latency was going —
  including the network hop, once we moved inference off the robot."
- "Converting to TensorRT with FP16 quantization cut inference time from
  X ms to Y ms, which mattered because it's the difference between the
  robot reacting in time and not."
- "We deliberately separated the safety-critical balance loop from the
  vision pipeline, so that neither GPU latency variance nor WiFi jitter
  could destabilize the robot — the balance loop simply continues on the
  last known command if a frame or a network packet is late."
- "When the original onboard-Jetson plan became impractical due to hardware
  availability, we re-architected to run inference on a laptop GPU instead,
  with the robot carrying only a WiFi camera — without touching a single
  line of the CUDA/TensorRT pipeline code, because we'd designed the
  interface between perception and control as a clean serial protocol from
  day one."
