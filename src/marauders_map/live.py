"""Headless detection + tracking loop, drives the web view's /api/state.

Runs in a background thread when `marauders serve --live`. Reads streams from
each camera in the loaded `house.yaml`, runs detection + per-camera tracker,
projects foot-points to the floor plan, fuses identities cross-camera, and
publishes the latest markers + trails into a `LiveState` for the HTTP layer
to read.

No cv2 windows: this is the difference between this loop and `app._run`.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from marauders_map.capture import ThreadedCapture
from marauders_map.classes import resolve_classes
from marauders_map.detector import make_detector
from marauders_map.house import House
from marauders_map.logger import TrackLogger
from marauders_map.pipeline import process_camera
from marauders_map.tracker_global import GlobalTracker, PersonMarker


class LiveState:
    """Thread-safe latest-state for the web layer to read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._markers: list[dict] = []
        self._trails: dict[int, list[tuple[float, float]]] = {}
        self._fps: float = 0.0
        self._last_update: float = 0.0
        self._running: bool = False
        self._error: str | None = None
        # Latest annotated frame per camera (BGR np.ndarray) for MJPEG streaming.
        self._frames: dict = {}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "markers": list(self._markers),
                "trails": {str(k): [list(p) for p in v]
                           for k, v in self._trails.items()},
                "fps": self._fps,
                "last_update": self._last_update,
                "running": self._running,
                "error": self._error,
            }

    def set(
        self,
        markers: list[dict],
        trails: dict[int, list[tuple[float, float]]],
        fps: float,
    ) -> None:
        with self._lock:
            self._markers = markers
            self._trails = trails
            self._fps = fps
            self._last_update = time.time()

    def set_running(self, running: bool, error: str | None = None) -> None:
        with self._lock:
            self._running = running
            self._error = error

    def set_frame(self, cam_id: str, frame) -> None:
        # Frame is mutated by the capture thread on next read; store a copy.
        with self._lock:
            self._frames[cam_id] = frame.copy()

    def latest_frame(self, cam_id: str):
        with self._lock:
            f = self._frames.get(cam_id)
            return None if f is None else f.copy()


def run_live_loop(
    house_path: Path,
    state: LiveState,
    *,
    calibration_path: Path | None = None,
    backend: str = "ultralytics",
    model: str = "yolov8n.pt",
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    device: str | None = None,
    imgsz: int = 640,
    classes: str = "person",
    max_distance_m: float = 1.5,
    max_gap_s: float = 3.0,
    trail_s: float = 3.0,
    use_appearance: bool = True,
    log_path: Path | None = None,
    names: dict[int, str] | None = None,
    target_fps: float = 15.0,
) -> None:
    """Run the headless pipeline forever, publishing to `state`. Daemon-friendly."""
    try:
        house = House.load(house_path, calibration_path=calibration_path)
    except Exception as e:
        state.set_running(False, f"failed to load house.yaml: {e}")
        return

    cameras = house.cameras
    if not cameras:
        state.set_running(False, "no cameras configured (use the editor to add some)")
        return

    captures = []
    detectors = []
    try:
        for cam in cameras:
            captures.append(ThreadedCapture(cam.source).start())
        for _ in cameras:
            detectors.append(make_detector(
                backend, model=model, tracker=tracker, conf=conf,
                device=device, imgsz=imgsz,
            ))
    except Exception as e:
        state.set_running(False, f"failed to start capture/detector: {e}")
        for c in captures:
            c.stop()
        for d in detectors:
            d.close()
        return

    target_classes = resolve_classes(classes)
    global_tracker = GlobalTracker(
        max_distance_m=max_distance_m, max_gap_s=max_gap_s,
        trail_s=trail_s, use_appearance=use_appearance,
    )
    track_logger = TrackLogger(log_path) if log_path else None
    state.set_running(True)

    fps_ema: float | None = None
    min_dt = 1.0 / max(1.0, target_fps)

    def _mtime(p: Path | None) -> float:
        try:
            return p.stat().st_mtime if (p and p.exists()) else 0.0
        except OSError:
            return 0.0

    house_mtime = _mtime(house_path)
    cal_mtime = _mtime(calibration_path)
    last_check = 0.0

    try:
        while True:
            t0 = time.time()

            # Hot-reload calibration / camera placement at most once per second.
            if t0 - last_check > 1.0:
                last_check = t0
                new_house_mt = _mtime(house_path)
                new_cal_mt = _mtime(calibration_path)
                if new_house_mt != house_mtime or new_cal_mt != cal_mtime:
                    try:
                        reloaded = House.load(
                            house_path, calibration_path=calibration_path,
                        )
                        new_by_id = {c.id: c for c in reloaded.cameras}
                        applied = 0
                        for cam in cameras:
                            nc = new_by_id.get(cam.id)
                            if nc is None:
                                continue
                            cam.calibration = nc.calibration
                            cam.position = nc.position
                            cam.heading = nc.heading
                            cam.fov = nc.fov
                            if nc.calibration.homography is not None:
                                applied += 1
                        # Use the reloaded house for room_at lookups; keep the
                        # original `cameras` list so capture/detector indices
                        # stay valid even if the new yaml added/removed cams.
                        house = reloaded
                        house_mtime = new_house_mt
                        cal_mtime = new_cal_mt
                        print(
                            f"[live] reloaded house ({applied}/{len(cameras)} "
                            f"cameras calibrated)",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[live] reload failed: {e}", flush=True)

            all_dets = []
            for cam, cap, det in zip(cameras, captures, detectors):
                frame = cap.latest()
                if frame is None:
                    continue
                annotated, dets = process_camera(cam, frame, det, target_classes)
                state.set_frame(cam.id, annotated)
                all_dets.extend(dets)

            markers_in = [
                PersonMarker(d.cam_id, d.local_id, d.floor_xy, d.embedding)
                for d in all_dets
            ]
            global_markers = global_tracker.update(t0, markers_in)
            trails = global_tracker.trails(t0)

            marker_dicts = []
            for m in global_markers:
                room = house.room_at(m.floor_xy)
                marker_dicts.append({
                    "global_id": m.global_id,
                    "name": (names or {}).get(m.global_id, f"P{m.global_id}"),
                    "cam_id": m.cam_id,
                    "x": m.floor_xy[0],
                    "y": m.floor_xy[1],
                    "room": room,
                })
            trail_pts = {gid: list(pts) for gid, pts in trails.items()}

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps
            state.set(marker_dicts, trail_pts, fps_ema)

            if track_logger is not None:
                track_logger.log(t0, global_markers, house)

            slack = min_dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    except Exception as e:
        state.set_running(False, f"live loop crashed: {e}")
    finally:
        for c in captures:
            c.stop()
        for d in detectors:
            d.close()
        if track_logger is not None:
            track_logger.close()
