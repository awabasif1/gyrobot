# ESP32-CAM bridge — flashing and first contact

The sketch in this folder turns an ESP32-CAM into a camera with a WiFi antenna
and a serial port. It knows nothing about balancing. See
[`../mcu/SERIAL_PROTOCOL.md`](../mcu/SERIAL_PROTOCOL.md) for what it speaks.

## Arduino IDE setup

1. **Board support**: File → Preferences → *Additional board manager URLs*:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
   Then Tools → Board → Boards Manager → install **esp32** by Espressif.

2. **Board settings** — these matter:

   | Setting | Value | Why |
   |---|---|---|
   | Board | AI Thinker ESP32-CAM | sets the pin map |
   | Partition Scheme | **Huge APP (3MB No OTA/1MB SPIFFS)** | the default is too small; the camera driver plus WiFi overflows it and you get a link error about a section not fitting |
   | PSRAM | **Enabled** | without it there is nowhere to hold two VGA JPEG buffers and the sketch falls back to QVGA |
   | Upload Speed | 460800 | drop to 115200 if uploads fail |

3. **Edit the top of the sketch**: `WIFI_SSID`, `WIFI_PASS`, and the static IP
   block. A fixed address is worth the two minutes — with DHCP the board's
   address changes and every command line on the laptop changes with it,
   usually mid-demo.

## Flashing

With the ESP32-CAM-MB baseboard (the one with the USB socket), just plug it in
and upload. Without it, you need a USB-serial adapter and must jumper
**GPIO0 to GND** before powering up to enter bootloader mode, then remove the
jumper and reset to run.

If upload fails with `Failed to connect to ESP32: Timed out waiting for packet
header`, it is nearly always GPIO0 not being grounded at reset, or a USB cable
that only carries power.

## First contact — before any robot code

Open the Serial Monitor at 115200 and reset the board. It prints its IP and the
exact flags to give the laptop. Then, from the laptop:

```bash
curl "http://<ip>/status"                        # uptime, RSSI, counters
curl "http://<ip>/cmd?lean=0.00&turn=0.00"       # expect: OK
```

and open `http://<ip>/` in a browser — you should see live video. Do this
**before** writing or running anything else. If the browser shows video and the
C++ pipeline cannot open the stream, the problem is that OpenCV was built
without FFMPEG, not the camera.

Give the laptop:

```bash
./build/reflex_cv --stream http://<ip>:81/stream --no-send
```

The stream is on **port 81**; commands are on port 80 and the pipeline finds
them itself. (Two HTTP servers, because an MJPEG handler never returns and
would otherwise block `/cmd` forever — the sketch's header comment explains
this in full.)

## Things that will go wrong

**`Camera init failed with error 0x105`** — almost always power, not software.
The camera draws ~250 mA in bursts during initialisation. Feed the board 5 V
from the UBEC, not from the Mega's 5 V pin, whose regulator browns out. Also
reseat the camera ribbon; it is easy to get half-latched.

**Stream stutters or stalls** — check `RSSI` on `/status`. Weaker than about
−70 dBm will stutter. The board's antenna is a PCB trace and the robot's
battery and motor wiring sit right next to it; moving the board a couple of
centimetres can change this a lot. Some modules have a solder jumper to switch
to an external antenna — if yours does and you have one, use it.

**High, variable command latency** — make sure `WiFi.setSleep(false)` is still
in the sketch. WiFi modem sleep is on by default, saves power the robot does not
need, and adds 100–200 ms of random latency to every request. It is the single
worst default for this application.

**The Mega sees nothing on the UART** — in order: common ground connected?
GPIO14 → Mega RX1 (pin 19)? Both at 115200? The level-shifting divider on the
*other* direction wired right? Flash `../mcu/uart_echo_test.ino` to the Mega
(it never touches the motors) and watch the Serial Monitor while you curl
`/cmd`.

**It worked and now the board reboots constantly** — look at the Serial Monitor
for a brownout message or a guru meditation backtrace. Repeated resets on a
robot are usually the motors browning out the shared 5 V rail; that is what the
separate UBEC feed is for.

## A note on the stock CameraWebServer example

The build guide suggests starting from the stock `CameraWebServer` example.
This sketch is a self-contained replacement rather than an edit of it: the
stock example spreads a web UI, a settings system and the streaming code across
several files, and almost all of it is irrelevant here. What remains is the
camera init, the MJPEG handler, and three small endpoints — about 300 lines you
can read in one sitting, which matters more than saving the typing.
