from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from marauders_map.capture import ThreadedCapture
from marauders_map.classes import class_name, resolve_classes
from marauders_map.detector import Detector, make_detector
from marauders_map.geometry import foot_point, project_point
from marauders_map.house import CameraConfig, House
from marauders_map.logger import TrackLogger
from marauders_map.tracker_global import GlobalMarker, GlobalTracker, PersonMarker
from marauders_map.viz import (
    draw_floorplan,
    draw_markers,
    draw_mischief_managed,
    draw_trails,
    headings_from_trails,
)


@dataclass
class _Detection:
    cam_id: str
    local_id: int
    cls: int
    bbox: tuple[float, float, float, float]
    floor_xy: tuple[float, float]
    embedding: np.ndarray | None = None


def _tile(frames: list[np.ndarray], cell: tuple[int, int] = (640, 480)) -> np.ndarray:
    w, h = cell
    n = len(frames)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = cv2.resize(f, (w, h))
    return grid


def _process_camera(
    cam: CameraConfig,
    frame: np.ndarray,
    detector: Detector,
    target_classes: set[int],
) -> tuple[np.ndarray, list[_Detection]]:
    tracked = detector.track(frame)
    annotated = tracked.annotated
    cv2.putText(annotated, cam.name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    detections: list[_Detection] = []
    H = cam.calibration.homography
    if H is not None:
        for d in tracked.detections:
            if d.cls not in target_classes:
                continue
            floor_xy = project_point(H, foot_point(d.bbox))
            detections.append(_Detection(
                cam.id, d.track_id, d.cls, d.bbox, floor_xy, d.embedding,
            ))
    return annotated, detections


def _overlay_global_ids(
    annotated: np.ndarray,
    cam_detections: list[_Detection],
    gid_lookup: dict[tuple[str, int], int],
    names: dict[int, str] | None,
) -> None:
    for d in cam_detections:
        gid = gid_lookup.get((d.cam_id, d.local_id))
        if gid is None:
            continue
        x1, y1, _, _ = d.bbox
        nm = (names or {}).get(gid, f"P{gid}")
        label = f"{nm} ({class_name(d.cls)})"
        cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def _parse_names(spec: str | None) -> dict[int, str]:
    """'Vlad,Hermione,Ron' → {1: 'Vlad', 2: 'Hermione', 3: 'Ron'}."""
    if not spec:
        return {}
    out: dict[int, str] = {}
    for i, name in enumerate(spec.split(","), start=1):
        n = name.strip()
        if n:
            out[i] = n
    return out


def _run(
    house: House | None,
    cameras: list[CameraConfig],
    backend: str,
    model_path: str,
    tracker: str,
    conf: float,
    device: str | None,
    imgsz: int,
    target_classes: set[int],
    gt_params: dict,
    names: dict[int, str],
    log_path: Path | None = None,
    plan_min_dt: float = 1.0 / 20.0,
) -> None:
    captures = [ThreadedCapture(cam.source).start() for cam in cameras]
    detectors: list[Detector] = [
        make_detector(
            backend, model=model_path, tracker=tracker, conf=conf,
            device=device, imgsz=imgsz,
        )
        for _ in cameras
    ]
    global_tracker = GlobalTracker(**gt_params) if house is not None else None
    track_logger = TrackLogger(log_path) if (log_path and house is not None) else None

    deadline = time.time() + 3.0
    while time.time() < deadline and any(c.latest() is None for c in captures):
        time.sleep(0.05)

    fps_ema: float | None = None
    cam_window = "home_marauders_map — cams"
    plan_window = "home_marauders_map — Marauder's Map"
    show_people = True
    last_plan_t = 0.0
    last_plan: np.ndarray | None = None

    try:
        while True:
            t0 = time.time()
            annotated_by_cam: dict[str, np.ndarray] = {}
            dets_by_cam: dict[str, list[_Detection]] = {}
            all_dets: list[_Detection] = []

            for cam, cap, det in zip(cameras, captures, detectors):
                frame = cap.latest()
                if frame is None:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, f"{cam.name}: no signal", (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    annotated_by_cam[cam.id] = blank
                    dets_by_cam[cam.id] = []
                    continue
                annotated, detections = _process_camera(cam, frame, det, target_classes)
                annotated_by_cam[cam.id] = annotated
                dets_by_cam[cam.id] = detections
                all_dets.extend(detections)

            global_markers: list[GlobalMarker] = []
            trails: dict[int, list[tuple[float, float]]] = {}
            if global_tracker is not None:
                markers_in = [
                    PersonMarker(d.cam_id, d.local_id, d.floor_xy, d.embedding)
                    for d in all_dets
                ]
                global_markers = global_tracker.update(t0, markers_in)
                trails = global_tracker.trails(t0)
                gid_lookup = {(g.cam_id, g.track_id): g.global_id for g in global_markers}
                for cam_id, dets in dets_by_cam.items():
                    _overlay_global_ids(annotated_by_cam[cam_id], dets, gid_lookup, names)

                if track_logger is not None and house is not None:
                    track_logger.log(t0, global_markers, house)

            grid = _tile([annotated_by_cam[c.id] for c in cameras])
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps
            cv2.putText(grid, f"{fps_ema:.1f} FPS (pipeline)",
                        (10, grid.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(cam_window, grid)

            if house is not None and (t0 - last_plan_t) >= plan_min_dt:
                plan = draw_floorplan(house)
                if show_people:
                    draw_trails(plan, house, trails)
                    headings = headings_from_trails(trails)
                    draw_markers(plan, house, global_markers,
                                 names=names, headings=headings)
                else:
                    draw_mischief_managed(plan)
                last_plan = plan
                last_plan_t = t0
            if last_plan is not None:
                cv2.imshow(plan_window, last_plan)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show_people = not show_people
                last_plan_t = 0.0  # force redraw next loop
    finally:
        for cap in captures:
            cap.stop()
        for det in detectors:
            det.close()
        if track_logger is not None:
            track_logger.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Home Marauder's Map — real-time person tracking on a floor plan",
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to house.yaml (enables multi-cam + floor plan + global IDs)")
    parser.add_argument("--calibration", type=Path, default=Path("config/calibration.yaml"),
                        help="Calibration file (applied if it exists)")
    parser.add_argument("--source", default=None,
                        help="Single-cam shortcut: camera index or video/rtsp path")
    parser.add_argument("--backend", default="ultralytics",
                        choices=["ultralytics", "hailo"],
                        help="Detection backend (default: ultralytics; hailo for Pi+Hailo deploy)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO weights (.pt for ultralytics, .hef for hailo)")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda (ultralytics only)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size. Lower = faster (try 416 or 320 on Mac)")
    parser.add_argument("--log", type=Path, default=None,
                        help="SQLite path to write per-person position history (requires --config)")
    parser.add_argument("--classes", default="person",
                        help="Comma-separated COCO classes to track. Examples: "
                             "'person', 'person,cat,dog', 'person,cat,dog,bird'. "
                             "Numeric ids also accepted.")
    parser.add_argument("--names", default=None,
                        help="Comma-separated names mapped by global id, e.g. "
                             "'Vlad,Hermione,Ron' → P1=Vlad, P2=Hermione, P3=Ron")
    parser.add_argument("--max-distance-m", type=float, default=1.5,
                        help="Max floor-plan distance (m) for linking a new detection "
                             "to an existing global track. Larger = more permissive "
                             "handoffs, more ID mixups")
    parser.add_argument("--max-gap-s", type=float, default=3.0,
                        help="Grace period (s) for reassociating a track across a "
                             "silent gap (e.g. walking between cameras via a door)")
    parser.add_argument("--trail-s", type=float, default=3.0,
                        help="How much recent history to draw as a trail per person")
    parser.add_argument("--no-reid", action="store_true",
                        help="Disable the appearance tiebreaker (debugging only)")
    parser.add_argument("--map-fps", type=float, default=20.0,
                        help="Cap floor-plan redraw FPS (cheaper than every frame)")
    args = parser.parse_args()

    if args.config and args.source is not None:
        parser.error("use --config or --source, not both")
    if args.log and not args.config:
        parser.error("--log requires --config (need a floor plan to log against)")

    if args.config:
        house = House.load(args.config, calibration_path=args.calibration)
        cameras = house.cameras
    else:
        src = args.source if args.source is not None else "0"
        src_val: int | str = int(src) if src.isdigit() else src
        house = None
        cameras = [CameraConfig(id="cam0", name="webcam", source=src_val)]

    target_classes = resolve_classes(args.classes)
    names = _parse_names(args.names)
    gt_params = {
        "max_distance_m": args.max_distance_m,
        "max_gap_s": args.max_gap_s,
        "trail_s": args.trail_s,
        "use_appearance": not args.no_reid,
    }

    plan_min_dt = 1.0 / max(1.0, args.map_fps)
    _run(
        house, cameras, args.backend, args.model, args.tracker, args.conf,
        args.device, args.imgsz, target_classes, gt_params, names,
        args.log, plan_min_dt,
    )


if __name__ == "__main__":
    main()
