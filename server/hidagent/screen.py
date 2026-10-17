"""The phone's screen, received over AirPlay mirroring by UxPlay, as JPEG frames."""

import asyncio
import hashlib
import logging
import os
import re
import subprocess
import threading
import time

log = logging.getLogger(__name__)


class Screen:
    """Runs UxPlay with a GStreamer sink that writes JPEGs into a pipe; keeps the latest one."""

    def __init__(self, name: str = "iPhone Agent", fps: int = 8, quality: int = 75):
        self.name, self.fps, self.quality = name, fps, quality
        self.frame: bytes | None = None
        self.updated = 0.0
        self._proc = None
        self._clients_at, self._clients_n = 0.0, None
        self._stopping = False
        self.source_epoch, self.source_key, self.source = 0, None, {}

    def start(self) -> None:
        threading.Thread(target=self._supervise, daemon=True).start()

    def stop(self) -> None:
        self._stopping = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)

    def _source_line(self, line):
        match = re.search(r"connection request from .+ \(([^)]+)\) with deviceID = (\S+)", line)
        if match:
            model, identity = match.groups()
            self.source_key = hashlib.sha256(identity.encode()).hexdigest()[:16]
            self.source_epoch += 1
            family = "iPad" if model.startswith("iPad") else "iPhone" if model.startswith("iPhone") else "Apple device"
            self.source = {"model": model, "family": family, "key": self.source_key}

    def _logs(self, process):
        for raw in process.stdout:
            line = raw.decode(errors="replace").strip()
            self._source_line(line)
            log.info("uxplay: %s", re.sub(r"deviceID = \S+", "deviceID = <redacted>", line))

    def _supervise(self) -> None:
        """Keep UxPlay running: it exits on some disconnects, so restart it every time."""
        while not self._stopping:
            read_fd, write_fd = os.pipe()
            sink = (f"videorate drop-only=true ! video/x-raw,framerate={self.fps}/1 ! "
                    f"jpegenc quality={self.quality} ! fdsink fd={write_fd}")
            self._proc = subprocess.Popen(
                ["uxplay", "-n", self.name, "-nh", "-a", "-reset", "0", "-vs", sink],
                pass_fds=(write_fd,), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            threading.Thread(target=self._logs, args=(self._proc,), daemon=True).start()
            os.close(write_fd)
            log.info("AirPlay receiver '%s' started (pid %d)", self.name, self._proc.pid)
            self._read(read_fd)  # returns when UxPlay closes the pipe
            code = self._proc.wait()
            self.frame = None
            log.warning("AirPlay receiver exited (code %s), restarting", code)
            time.sleep(1)

    @property
    def live(self) -> bool:
        running = self._proc is not None and self._proc.poll() is None
        if not running or self.frame is None:
            return False
        clients = self._clients()
        if clients is not None:  # Linux: is a phone connected to UxPlay right now?
            return clients > 0
        return time.time() - self.updated < 90  # elsewhere: AirPlay sends no frames while the screen is still

    def _clients(self) -> int | None:
        """Established TCP connections of the UxPlay process (a mirroring phone keeps some open).

        Read from /proc, cached for a second; None where /proc is not available (e.g. macOS).
        """
        if time.time() - self._clients_at < 1:
            return self._clients_n
        try:
            fds = os.listdir(f"/proc/{self._proc.pid}/fd")
            inodes = set()
            for fd in fds:
                link = os.readlink(f"/proc/{self._proc.pid}/fd/{fd}")
                if link.startswith("socket:["):
                    inodes.add(link[8:-1])
            n = 0
            for table in ("/proc/net/tcp", "/proc/net/tcp6"):
                with open(table) as f:
                    for row in f.readlines()[1:]:
                        cols = row.split()
                        n += cols[3] == "01" and cols[9] in inodes  # 01 = ESTABLISHED
        except OSError:
            n = None
        self._clients_at, self._clients_n = time.time(), n
        return n

    @property
    def geometry(self) -> tuple[int, int] | None:
        """The mirrored frame size; no device model or portrait assumption."""
        f = self.frame or b""
        i = f.find(b"\xff\xc0")  # baseline start-of-frame: ... height(2) width(2)
        if i < 0 or len(f) < i + 9:
            return None
        return int.from_bytes(f[i + 7:i + 9], "big"), int.from_bytes(f[i + 5:i + 7], "big")

    @property
    def aspect(self) -> float:
        size = self.geometry
        return size[0] / size[1] if size else 9 / 19.5

    async def settled(self, quiet: float = 0.35, least: float = 0.25, most: float = 4) -> None:
        """Wait until the screen stops changing (AirPlay only sends frames on change)."""
        started = time.time()
        await asyncio.sleep(least)
        while time.time() - self.updated < quiet and time.time() - started < most:
            await asyncio.sleep(0.1)

    async def fresh(self, since: float, timeout: float = 0.8) -> bytes | None:
        """The first frame received after `since`, or None if none arrives in time."""
        deadline = time.time() + timeout
        while self.updated <= since:
            if time.time() > deadline:
                return None
            await asyncio.sleep(0.02)
        return self.frame

    def _read(self, fd: int) -> None:
        buf = b""
        with os.fdopen(fd, "rb", buffering=0) as pipe:
            while chunk := pipe.read(1 << 16):
                buf += chunk
                while (start := buf.find(b"\xff\xd8")) >= 0 and (end := buf.find(b"\xff\xd9", start)) >= 0:
                    self.frame, self.updated = buf[start:end + 2], time.time()
                    buf = buf[end + 2:]
