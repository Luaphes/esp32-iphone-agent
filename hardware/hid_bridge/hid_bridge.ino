// HID bridge: an ESP32-S3 that shows up as a USB mouse + keyboard and takes
// line commands over TCP (Wi-Fi). It holds no logic; the server decides.
//
// Protocol (one command per line, the board answers "ok" or "err <why>"):
//   ping | home | move <dx> <dy> | press | release | click
//   scroll <v> <h> | type <ascii text> | key <combo, e.g. cmd+h, enter>
//   homescreen (USB consumer Home key; distinct from pointer-homing `home`)

#include <WiFi.h>
#include <esp_task_wdt.h>
#include "USB.h"
#include "USBHIDMouse.h"
#include "USBHIDKeyboard.h"
#include "USBHIDConsumerControl.h"
#include "tusb.h"
#include "secrets.h"  // WIFI_SSID, WIFI_PASS, SERVER_HOST, SERVER_PORT

const int STEP = 10;      // max pointer delta per report while moving
const int STEP_MS = 8;    // report interval; constant pace keeps iOS acceleration predictable
const int HOME_REPORTS = 50;
#ifndef DEBUG_SERIAL
#define DEBUG_SERIAL 0  // 1: echo every command over USB serial (needs CDCOnBoot=cdc)
#endif

const int WATCHDOG_MS = 8000;  // a hung loop (e.g. blocked USB) reboots the board instead of staying offline
// Apple Accessory Design Guidelines: Consumer page 0x0C, Menu is the Home button.
// AC Home (CONSUMER_CONTROL_HOME, 0x0223) is a different key and does not go Home on iPhone.
const uint16_t IOS_HOME_USAGE = 0x0040;

USBHIDMouse mouse;
USBHIDKeyboard keyboard;
USBHIDConsumerControl consumer;
WiFiClient server;

void movePaced(long dx, long dy) {
  while (dx || dy) {
    int sx = constrain(dx, -STEP, STEP), sy = constrain(dy, -STEP, STEP);
    mouse.move(sx, sy);
    dx -= sx;
    dy -= sy;
    delay(STEP_MS);
  }
}

uint8_t keyCode(const String &k) {
  if (k == "enter") return KEY_RETURN;
  if (k == "esc") return KEY_ESC;
  if (k == "tab") return KEY_TAB;
  if (k == "backspace") return KEY_BACKSPACE;
  if (k == "space") return ' ';
  if (k == "up") return KEY_UP_ARROW;
  if (k == "down") return KEY_DOWN_ARROW;
  if (k == "left") return KEY_LEFT_ARROW;
  if (k == "right") return KEY_RIGHT_ARROW;
  if (k == "cmd") return KEY_LEFT_GUI;
  if (k == "ctrl") return KEY_LEFT_CTRL;
  if (k == "alt") return KEY_LEFT_ALT;
  if (k == "shift") return KEY_LEFT_SHIFT;
  return k.length() == 1 ? k[0] : 0;
}

bool pressCombo(String combo) {
  combo += '+';
  for (int i; (i = combo.indexOf('+')) >= 0; combo = combo.substring(i + 1)) {
    uint8_t c = keyCode(combo.substring(0, i));
    if (!c) { keyboard.releaseAll(); return false; }
    keyboard.press(c);
  }
  delay(30);
  keyboard.releaseAll();
  return true;
}

bool pressConsumer(uint16_t usage) {
  keyboard.releaseAll();
  bool pressed = consumer.press(usage);
  delay(100);
  bool released = consumer.release();
  return pressed && released;
}

// A locked or sleeping phone suspends USB; sending HID reports then can stall. Ask it to wake up instead,
// and report "asleep" right away if it does not, so the board never hangs.
bool usbReady() {
  if (tud_suspended()) {
    tud_remote_wakeup();
    for (int i = 0; i < 50 && tud_suspended(); i++) delay(10);
  }
  return tud_mounted() && !tud_suspended();
}

String run(const String &line) {
  int sp = line.indexOf(' ');
  String cmd = sp < 0 ? line : line.substring(0, sp);
  String arg = sp < 0 ? "" : line.substring(sp + 1);
  long a = 0, b = 0;
  sscanf(arg.c_str(), "%ld %ld", &a, &b);

  if (cmd == "ping") return "ok";
  if (!usbReady()) return "err phone usb asleep (unlock the phone)";
  if (cmd == "home") {  // slam the pointer into the top-left corner
    for (int i = 0; i < HOME_REPORTS; i++) { mouse.move(-127, -127); delay(2); }
    return "ok";
  }
  if (cmd == "homescreen") {
    mouse.release(MOUSE_LEFT);
    return pressConsumer(IOS_HOME_USAGE) ? "ok" : "err homescreen USB report failed";
  }
  if (cmd == "move") { movePaced(a, b); return "ok"; }
  if (cmd == "press") { mouse.press(MOUSE_LEFT); return "ok"; }
  if (cmd == "release") { mouse.release(MOUSE_LEFT); return "ok"; }
  if (cmd == "click") { mouse.press(MOUSE_LEFT); delay(60); mouse.release(MOUSE_LEFT); return "ok"; }
  if (cmd == "scroll") { mouse.move(0, 0, a, b); return "ok"; }
  if (cmd == "type") {
    // Validate the entire input before any keypress: UTF-8 bytes can become modifier/key codes.
    for (size_t i = 0; i < arg.length(); i++) {
      uint8_t c = arg[i];
      if (c < 32 || c >= 127) return "err type requires printable ASCII; compose Chinese on the server";
    }
    for (size_t i = 0; i < arg.length(); i++) {
      if (!keyboard.write((uint8_t)arg[i])) { keyboard.releaseAll(); return "err type USB report failed"; }
      delay(10);
      esp_task_wdt_reset();
    }
    return "ok";
  }
  if (cmd == "key") {
    // Apple Consumer-page controls: input-source selector and onscreen keyboard toggle.
    if (arg == "globe") return pressConsumer(0x029D) ? "ok" : "err globe USB report failed";
    if (arg == "keyboard") return pressConsumer(0x01AE) ? "ok" : "err keyboard USB report failed";
    return pressCombo(arg) ? "ok" : "err unknown key";
  }
  return "err unknown command";
}

const char *resetReason() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON: return "power-on";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_TASK_WDT: case ESP_RST_INT_WDT: case ESP_RST_WDT: return "watchdog";
    case ESP_RST_PANIC: return "crash";
    case ESP_RST_SW: return "software";
    case ESP_RST_USB: return "usb";
    default: return "other";
  }
}

void setup() {
#if DEBUG_SERIAL
  Serial.begin(115200);
#endif
  mouse.begin();
  keyboard.begin();
  consumer.begin();
  USB.productName("HID Bridge");
  USB.begin();
  esp_task_wdt_config_t wdt = {.timeout_ms = WATCHDOG_MS, .idle_core_mask = 0, .trigger_panic = true};
  esp_task_wdt_reconfigure(&wdt);
  esp_task_wdt_add(NULL);
  WiFi.setHostname("hid-bridge");
  WiFi.setSleep(false);  // modem sleep adds ~100 ms per command
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void loop() {
  esp_task_wdt_reset();
  if (WiFi.status() != WL_CONNECTED) { mouse.release(MOUSE_LEFT); delay(50); return; }  // auto-reconnects
  if (!server.connected()) {
    mouse.release(MOUSE_LEFT);  // a gesture cut off mid-way must not leave the button held (long press)
    keyboard.releaseAll();
    consumer.release();
    if (!server.connect(SERVER_HOST, SERVER_PORT, 1000)) { delay(200); return; }  // retry fast: show up at once
    server.setNoDelay(true);
    server.printf("hello %s %lu\n", resetReason(), millis());  // why the board last restarted, and uptime
#if DEBUG_SERIAL
    Serial.printf("connected to %s:%d as %s after %lu ms\n", SERVER_HOST, SERVER_PORT,
                  WiFi.localIP().toString().c_str(), millis());
#endif
  }
  if (server.available()) {
    String line = server.readStringUntil('\n');
    line.trim();
    if (!line.length()) return;
    unsigned long started = millis();
    String reply = run(line);
    server.print(reply + "\n");
#if DEBUG_SERIAL
    Serial.printf("%s -> %s (%lu ms)\n", line.c_str(), reply.c_str(), millis() - started);
#endif
  }
}
