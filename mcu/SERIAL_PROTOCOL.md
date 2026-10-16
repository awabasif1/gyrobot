# Serial protocol

The contract between the three boards. All three implementations must agree
with this document; if you change anything here, change all three and re-run
`make -C test`.

```
  laptop  ──HTTP/WiFi──▶  ESP32-CAM  ──UART 115200──▶  Arduino Mega
    ▲                                                        │
    └────────────── HTTP ◀── cached ◀── UART ────────────────┘
```

Two hops, two formats. The UART format is the real protocol; the HTTP layer
exists only because the laptop cannot plug into a UART from across the room.

---

## Hop 1 — laptop → ESP32-CAM (HTTP GET)

```
GET /cmd?lean=<degrees>&turn=<ratio>
```

| Parameter | Type | Range | Meaning |
|---|---|---|---|
| `lean` | float, 2 dp | −5.0 … +5.0 | Offset added to the balance setpoint, in **degrees**. Positive = lean forward = drive forward. |
| `turn` | float, 2 dp | −1.0 … +1.0 | Differential steering, **dimensionless**. Positive = turn right (clockwise seen from above). |

Response: `200 OK` with the body `OK`. Any other status means the command did
not reach the UART.

Chosen as a GET with query parameters for one reason: it can be tested by hand,
before any of the robot's code exists.

```bash
curl "http://192.168.1.50/cmd?lean=0.00&turn=0.50"    # gentle right turn
curl "http://192.168.1.50/cmd?lean=0.00&turn=0.00"    # stop steering
```

Implemented by `cmd_handler()` in `esp32cam/CameraWebServer_uart_bridge.ino`,
called from `Esp32Link::send_command_now()` in `src/esp32_client.cpp`.

### Other endpoints on the bridge

| Endpoint | Purpose |
|---|---|
| `GET /` | HTML page showing the live stream — check the camera from a phone |
| `GET :81/stream` | MJPEG stream. **Port 81**, not 80 — see below |
| `GET /telemetry` | Last line the Mega sent, plus its age |
| `GET /status` | Plain-text diagnostics: uptime, RSSI, frame/command counters |
| `GET /res?size=` | `qvga`/`vga`/`svga`/`xga` — restart the laptop pipeline after |

**The stream is on port 81 and everything else is on port 80.** An MJPEG
handler never returns, and `esp_http_server` serves one request at a time per
instance, so sharing a port would mean `/cmd` is queued behind the stream
forever. Two server instances, two ports, two tasks.

---

## Hop 2 — ESP32-CAM → Mega (UART, 115200 8N1)

The line the Mega actually parses:

```
L:<lean>,T:<turn>\n
```

- ASCII, `\n`-terminated. `\r` is tolerated and ignored.
- Both fields are required, in this order.
- Two decimal places by convention; the parser accepts any number of digits.
- Exponent notation (`1e2`) is **rejected** — the sender never produces it, so
  seeing it means the line is corrupt.
- Keys are case-insensitive; spaces around the comma are allowed.
- Maximum line length 31 characters. Longer lines are dropped, not truncated.

Examples:

```
L:0.00,T:0.00      robot stands still
L:2.50,T:0.00      lean forward 2.5 degrees (drive forward)
L:0.00,T:-0.75     turn left, hard
L:-1.20,T:0.30     back off slightly while turning right
```

The ESP32 **re-formats** the numbers rather than passing the query string
through. A malformed query therefore cannot inject extra characters — or, worse,
a newline — into the Mega's input stream.

### Return path — Mega → ESP32-CAM (telemetry)

```
A:<angle>,F:<flag>\n
```

| Field | Meaning |
|---|---|
| `A` | Current filtered tilt angle, degrees. Positive = leaning forward. |
| `F` | `0` = motors armed and balancing, `1` = disarmed (fallen, or not yet armed) |

Sent at 20 Hz by the Mega. The ESP32 caches the most recent line and serves it
from `/telemetry` with an `AGE:<ms>` field appended, so the laptop can tell a
stale value from a live one:

```
A:1.37,F:0,AGE:42
```

---

## Timing and failure behaviour

This is the part that matters most, and it is the reason the system is split
across three boards at all.

| Link | Rate | If it stops |
|---|---|---|
| Mega balance loop | 200 Hz | Nothing else matters; this must never stall |
| Laptop → robot commands | ≤ 20 Hz | Mega zeroes the vision command after 400 ms |
| Mega → laptop telemetry | 20 Hz | Laptop shows the reading as stale; nothing else changes |
| Camera stream | 15–30 fps | Shaper decays the command to zero; Mega keeps balancing |

**The balance loop never waits for any of this.** It reads whatever command
arrived most recently and runs regardless. There are three independent
safeguards, deliberately layered, because the failure they guard against — the
robot driving off while nobody is in control of it — is the one that breaks
hardware:

1. **Laptop** (`CommandShaper`): when the target is lost, hold the last command
   for 350 ms, then ramp to zero over 400 ms.
2. **Mega** (`CMD_TIMEOUT_MS`): if no valid line arrives for 400 ms, the vision
   command is forced to zero, whatever was last received. This covers the
   laptop crashing, the WiFi dropping, and the UART cable falling out.
3. **Laptop shutdown**: Ctrl-C sends `L:0.00,T:0.00` before exiting. Without
   this, quitting the program leaves the Mega acting on the last command
   forever.

Note that 2 does not depend on 1, and 3 does not depend on either. The Mega's
timeout is the authoritative one, because it is the only board that still works
when the other two are gone.

---

## Electrical

**The Mega → ESP32 direction needs a level shifter.** The Mega's TX idles at
5 V; the ESP32's pins are 3.3 V. A two-resistor divider is enough — 1 kΩ from
Mega TX to ESP32 RX, 2 kΩ from ESP32 RX to GND. The other direction needs
nothing: 3.3 V is a valid HIGH for a 5 V AVR input.

```
  Mega TX1 (18) ──[1k]──┬──▶ ESP32 RX (GPIO13)
                        │
                       [2k]
                        │
                       GND

  Mega RX1 (19) ◀───────────── ESP32 TX (GPIO14)
  GND ────────────────────────── GND          ← common ground is mandatory
```

Pins 18/19 are the Mega's `Serial1`. `Serial` (USB) stays free for debugging —
that separation is the main reason this project uses a Mega rather than an Uno.

On the ESP32-CAM, GPIO13 and GPIO14 are SD-card pins, free whenever the SD card
is unused, and unlike GPIO12 and GPIO15 they are not boot strapping pins — so
whatever the Mega happens to be driving at power-on cannot change how the ESP32
boots.

---

## Testing each hop independently

Do these in order. Each one works without the next stage existing.

**Hop 1 alone** — ESP32 flashed, Mega not even connected:
```bash
curl "http://<esp32-ip>/status"
curl "http://<esp32-ip>/cmd?lean=1.5&turn=-0.25"   # expect: OK
```

**Hop 2 alone** — flash `mcu/uart_echo_test.ino` to the Mega (it never touches
the motors), open the Serial Monitor at 115200, then repeat the curl above. The
line should appear in the monitor and be marked `-> OK`.

**Return path** — with the echo sketch still running:
```bash
curl "http://<esp32-ip>/telemetry"    # expect a sweeping angle, AGE under ~100
```

**Full pipeline, robot inert** — run the laptop side with `--no-send`, confirm
the commands printed look sensible as you move in front of the camera, then
drop the flag.

Only after all four pass should the balance firmware go on the Mega.
