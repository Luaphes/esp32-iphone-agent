#!/usr/bin/env bash
# Build and flash the HID bridge (Arduino ESP32 3.x).
# Usage: hardware/flash.sh [port] [--debug] [--build-only]
# A running bridge is a TinyUSB device, so it is first kicked into the ROM
# bootloader with a 1200-baud touch; a blank board is flashed directly.
set -euo pipefail
cd "$(dirname "$0")"
FQBN=${ESP32_FQBN:-"esp32:esp32:esp32s3:USBMode=default,CDCOnBoot=cdc,FlashSize=4M,PartitionScheme=default"}
CLI=(arduino-cli)
CONFIG=${ARDUINO_CONFIG_FILE:-../.tools/arduino-cli.yaml}
if [[ -f $CONFIG ]]; then CLI+=(--config-file "$CONFIG"); fi
PORT="" BUILD_ONLY=0
FLAGS=(--build-property "compiler.cpp.extra_flags=")
for arg in "$@"; do
  case "$arg" in
    --debug) FLAGS=(--build-property "compiler.cpp.extra_flags=-DDEBUG_SERIAL=1") ;;
    --build-only) BUILD_ONLY=1 ;;
    --*) echo "Unknown option: $arg" >&2; exit 1 ;;
    *) [[ -z $PORT ]] || { echo "Specify only one serial port" >&2; exit 1; }; PORT=$arg ;;
  esac
done
port() {
  local candidates=() p
  for p in /dev/cu.usbmodem* /dev/ttyACM*; do [[ ! -e $p ]] || candidates+=("$p"); done
  [[ ${#candidates[@]} -eq 1 ]] || return 1
  echo "${candidates[0]}"
}

"${CLI[@]}" compile -b "$FQBN" "${FLAGS[@]}" --output-dir build hid_bridge
(( BUILD_ONLY )) && exit 0
PORT=${PORT:-$(port)}
[[ -e $PORT ]] || { echo "Connect one board or specify its serial port" >&2; exit 1; }
OLD=$PORT
stty -f "$PORT" 1200 2>/dev/null || stty -F "$PORT" 1200 2>/dev/null || true
sleep 1
PORT=""
for _ in $(seq 20); do
  if [[ -e $OLD ]]; then PORT=$OLD; break; fi
  if PORT=$(port); then break; fi
  sleep 0.5
done
[[ -n $PORT && -e $PORT ]] || { echo "Bootloader port is missing or ambiguous; nothing was flashed" >&2; exit 1; }
"${CLI[@]}" upload -b "$FQBN" -p "$PORT" --input-dir build hid_bridge
