"""Threaded frame reader — decouples camera I/O from the detection loop."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class ThreadedCapture:
    """Background-thread wrapper around cv2.VideoCapture.

    Always exposes the most-recent frame via `latest()` without blocking.
    For local video files, loops back to the start on EOF.
    """

    def __init__(self, source: int | str, loop: bool = True):
        self.source = source
        self.loop = loop
        self._is_file = isinstance(source, str) and "://" not in source
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "ThreadedCapture":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            hint = ""
            if self.source == 0:
                hint = (" — on macOS make sure your terminal has Camera "
                        "permission in System Settings → Privacy & Security → Camera")
            elif isinstance(self.source, str) and self.source.startswith("rtsp"):
                hint = " — check the RTSP URL and that the camera is reachable on the network"
            elif isinstance(self.source, str):
                hint = " — check the file exists and is a readable video"
            raise RuntimeError(f"Could not open source: {self.source}{hint}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self._is_file and self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
